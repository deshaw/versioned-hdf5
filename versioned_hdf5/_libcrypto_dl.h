// Runtime resolution of the OpenSSL EVP entry points used by hash.pyx.
//
// hash.pyx must not link against libcrypto. A NEEDED entry pins a concrete soname
// (libcrypto.so.3, libcrypto-3-x64.dll, ...) into hash.so, so a binary built on one
// distribution cannot be imported on another, and building would need the OpenSSL
// development headers. Instead we resolve the handful of functions we call out of the
// libcrypto shared library that the process has already mapped - CPython's _hashlib
// loads one - and reuse it, so there is normally only one copy of OpenSSL in the
// process (a CPython whose _hashlib links OpenSSL statically, or has no _hashlib at
// all, may still end up with a second copy mapped from a well-known soname).

#pragma once

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <wchar.h>
#else
#include <dlfcn.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <link.h>
#endif

// dl_iterate_phdr enumerates the already-loaded ELF objects. glibc only declares it
// under _GNU_SOURCE, which CPython's pyconfig.h defines (Python.h is always included
// before this header); musl and other libcs declare it unconditionally.
#if defined(__linux__) && (!defined(__GLIBC__) || defined(__USE_GNU))
#define VH_HAVE_DL_ITERATE_PHDR 1
#endif

// dladdr() lives in <dlfcn.h> too; glibc declares it under _GNU_SOURCE like
// dl_iterate_phdr, Apple and musl unconditionally.
#if !defined(_WIN32) && (!defined(__GLIBC__) || defined(__USE_GNU))
#define VH_HAVE_DLADDR 1
#endif

// Opaque handle types. We only ever pass pointers around, so the definitions are
// irrelevant; the tags deliberately do not match OpenSSL's.
typedef struct vh_EVP_MD_CTX vh_EVP_MD_CTX;
typedef struct vh_EVP_MD vh_EVP_MD;
typedef struct vh_ENGINE vh_ENGINE;

// The resolved entry points; NULL until vh_libcrypto_init() succeeds. Signatures are
// those of <openssl/evp.h>.
static vh_EVP_MD_CTX *(*vh_EVP_MD_CTX_new)(void) = NULL;
static void (*vh_EVP_MD_CTX_free)(vh_EVP_MD_CTX *ctx) = NULL;
static const vh_EVP_MD *(*vh_EVP_sha256)(void) = NULL;
static int (*vh_EVP_DigestInit_ex)(
    vh_EVP_MD_CTX *ctx, const vh_EVP_MD *type, vh_ENGINE *impl) = NULL;
static int (*vh_EVP_DigestUpdate)(
    vh_EVP_MD_CTX *ctx, const void *d, size_t cnt) = NULL;
static int (*vh_EVP_DigestFinal_ex)(
    vh_EVP_MD_CTX *ctx, unsigned char *md, unsigned int *s) = NULL;

// OpenSSL_version(OPENSSL_VERSION). Not an error if absent (OpenSSL 1.0.x has no such
// symbol); it only feeds the informational vh_libcrypto_get_version().
static const char *(*vh_OpenSSL_version)(int type) = NULL;

#define VH_LIBCRYPTO_PATH_MAX 1024
#define VH_LIBCRYPTO_VERSION_MAX 256
#define VH_LIBCRYPTO_ERROR_MAX 512

static char vh_libcrypto_path[VH_LIBCRYPTO_PATH_MAX];
static char vh_libcrypto_version[VH_LIBCRYPTO_VERSION_MAX];
static char vh_libcrypto_error[VH_LIBCRYPTO_ERROR_MAX];

// 0 = not initialised yet, 1 = success, -1 = failure.
static int vh_libcrypto_state = 0;

#ifdef _WIN32
typedef HMODULE vh_lib_handle;
#else
typedef void *vh_lib_handle;
#endif

static void *vh_libcrypto_symbol(vh_lib_handle handle, const char *name);
static int vh_libcrypto_has_evp(vh_lib_handle handle);
static vh_lib_handle vh_libcrypto_validate(vh_lib_handle handle);

static void vh_libcrypto_fail(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(vh_libcrypto_error, sizeof(vh_libcrypto_error), fmt, args);
    va_end(args);
    vh_libcrypto_error[sizeof(vh_libcrypto_error) - 1] = '\0';
}

// Record the library path, failing (instead of silently truncating) if it does not fit.
static int vh_libcrypto_set_path(const char *path) {
    int n = snprintf(vh_libcrypto_path, sizeof(vh_libcrypto_path), "%s", path);
    if (n < 0 || (size_t)n >= sizeof(vh_libcrypto_path)) {
        vh_libcrypto_fail("libcrypto path longer than %d bytes",
                          (int)VH_LIBCRYPTO_PATH_MAX);
        return -1;
    }
    return 0;
}

#ifndef _WIN32

// Basename of a path, recognising both POSIX and Windows separators.
static const char *vh_libcrypto_basename(const char *path) {
    const char *base = path;
    const char *p;
    for (p = path; *p != '\0'; p++) {
        if (*p == '/' || *p == '\\')
            base = p + 1;
    }
    return base;
}

static int vh_libcrypto_starts_with(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

#ifdef __APPLE__
static int vh_libcrypto_ends_with(const char *s, const char *suffix) {
    size_t s_len = strlen(s);
    size_t suffix_len = strlen(suffix);
    return s_len >= suffix_len && strcmp(s + s_len - suffix_len, suffix) == 0;
}
#endif

// Whether a loaded object's file name is libcrypto on this platform. On Linux this
// matches libcrypto.so, libcrypto.so.1.1 and libcrypto.so.3 alike.
static int vh_libcrypto_is_libcrypto(const char *path) {
    const char *base = vh_libcrypto_basename(path);
#ifdef __APPLE__
    return vh_libcrypto_starts_with(base, "libcrypto")
        && vh_libcrypto_ends_with(base, ".dylib");
#else
    return vh_libcrypto_starts_with(base, "libcrypto.so");
#endif
}

// Handle to an already-mapped library. RTLD_NOLOAD asks the dynamic loader to fail
// rather than map a second copy; the plain dlopen() fallback still resolves to the
// existing mapping, as the loader deduplicates by file name. The handle is
// deliberately never dlclose()d: the library belongs to the process, not to us.
static vh_lib_handle vh_libcrypto_open(const char *name) {
    vh_lib_handle handle;

    if (name == NULL || name[0] == '\0')
        return NULL;
#ifdef RTLD_NOLOAD
    handle = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
    if (handle != NULL)
        return handle;
#endif
    return dlopen(name, RTLD_LAZY);
}

#if defined(VH_HAVE_DL_ITERATE_PHDR)

// Copy candidate names while dl_iterate_phdr holds the loader lock, then open and
// validate them after it returns. Calling dlopen/dlsym from the callback can deadlock.
#define VH_LIBCRYPTO_MAX_CANDIDATES 16
typedef struct {
    char paths[VH_LIBCRYPTO_MAX_CANDIDATES][VH_LIBCRYPTO_PATH_MAX];
    size_t count;
} vh_libcrypto_candidates;

static int vh_libcrypto_phdr_callback(
    struct dl_phdr_info *info, size_t size, void *data) {
    vh_libcrypto_candidates *candidates = (vh_libcrypto_candidates *)data;
    char *dst;
    int n;

    (void)size;
    // dlpi_name is empty for the main executable
    if (info->dlpi_name == NULL || !vh_libcrypto_is_libcrypto(info->dlpi_name))
        return 0;
    if (candidates->count >= VH_LIBCRYPTO_MAX_CANDIDATES)
        return 0;

    dst = candidates->paths[candidates->count];
    n = snprintf(dst, VH_LIBCRYPTO_PATH_MAX, "%s", info->dlpi_name);
    if (n >= 0 && (size_t)n < VH_LIBCRYPTO_PATH_MAX)
        candidates->count++;
    return 0;
}

static vh_lib_handle vh_libcrypto_find_loaded(void) {
    vh_libcrypto_candidates candidates;
    size_t i;

    vh_libcrypto_path[0] = '\0';
    memset(&candidates, 0, sizeof(candidates));
    dl_iterate_phdr(vh_libcrypto_phdr_callback, &candidates);
    for (i = 0; i < candidates.count; i++) {
        vh_lib_handle handle;
        if (vh_libcrypto_set_path(candidates.paths[i]) != 0)
            continue;
        handle = vh_libcrypto_validate(vh_libcrypto_open(candidates.paths[i]));
        if (handle != NULL)
            return handle;
    }
    vh_libcrypto_path[0] = '\0';
    return NULL;
}

#elif defined(__APPLE__)

static vh_lib_handle vh_libcrypto_find_loaded(void) {
    uint32_t i;
    uint32_t nimages = _dyld_image_count();

    vh_libcrypto_path[0] = '\0';
    for (i = 0; i < nimages; i++) {
        const char *name = _dyld_get_image_name(i);
        vh_lib_handle handle;
        if (name == NULL || !vh_libcrypto_is_libcrypto(name))
            continue;
        if (vh_libcrypto_set_path(name) != 0)
            continue;
        handle = vh_libcrypto_validate(vh_libcrypto_open(name));
        if (handle != NULL)
            return handle;
    }
    vh_libcrypto_path[0] = '\0';
    return NULL;
}

#else

// No way to enumerate the loaded objects on this libc: rely on the known sonames.
static vh_lib_handle vh_libcrypto_find_loaded(void) {
    return NULL;
}

#endif  // VH_HAVE_DL_ITERATE_PHDR

#else  // _WIN32

// Module names on Windows are case-insensitive (LibCrypto-3.dll == libcrypto-3.dll), so
// compare without regard to case. Only names that really start with "libcrypto" match:
// the prefix check rejects e.g. "LCrypto.dll".
static int vh_libcrypto_is_libcrypto_w(const wchar_t *name) {
    static const wchar_t prefix[] = L"libcrypto";
    static const wchar_t suffix[] = L".dll";
    const size_t prefix_len = sizeof(prefix) / sizeof(prefix[0]) - 1;
    const size_t suffix_len = sizeof(suffix) / sizeof(suffix[0]) - 1;
    size_t len = wcslen(name);

    return len >= prefix_len + suffix_len
        && _wcsnicmp(name, prefix, prefix_len) == 0
        && _wcsicmp(name + len - suffix_len, suffix) == 0;
}

// Overwrite the recorded library path with the full path of a loaded module: the
// Win32 module APIs only know bare module names.
static void vh_libcrypto_record_module_path(vh_lib_handle handle) {
    wchar_t wide[VH_LIBCRYPTO_PATH_MAX];
    char utf8[VH_LIBCRYPTO_PATH_MAX];
    DWORD n = GetModuleFileNameW(handle, wide, (DWORD)VH_LIBCRYPTO_PATH_MAX);

    if (n == 0 || n >= (DWORD)VH_LIBCRYPTO_PATH_MAX - 1)
        return;
    wide[n] = L'\0';
    if (WideCharToMultiByte(
            CP_UTF8, 0, wide, -1, utf8, (int)sizeof(utf8), NULL, NULL) == 0)
        return;
    vh_libcrypto_set_path(utf8);
}

// GetModuleHandleExW() only ever returns already-loaded modules, which is exactly what
// we want. Unlike GetModuleHandleW(), it also increments the module reference count;
// we deliberately never release a successfully selected handle, so the DLL cannot be
// unloaded while our function pointers are in use.
static vh_lib_handle vh_libcrypto_open(const char *name) {
    wchar_t wide[VH_LIBCRYPTO_PATH_MAX];
    HMODULE handle = NULL;

    if (name == NULL || name[0] == '\0')
        return NULL;
    if (MultiByteToWideChar(
            CP_UTF8, 0, name, -1, wide, (int)VH_LIBCRYPTO_PATH_MAX) == 0)
        return NULL;
    if (!GetModuleHandleExW(0, wide, &handle))
        return NULL;
    return handle;
}

// Retain a module found through Toolhelp enumeration before the snapshot is closed.
// FROM_ADDRESS makes GetModuleHandleExW() operate on the module containing the handle's
// base address, rather than looking up a potentially ambiguous module name.
static vh_lib_handle vh_libcrypto_retain(vh_lib_handle handle) {
    HMODULE retained = NULL;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)handle, &retained))
        return NULL;
    return retained;
}

static vh_lib_handle vh_libcrypto_find_loaded(void) {
    HANDLE snapshot;
    MODULEENTRY32W entry;
    vh_lib_handle handle = NULL;
    int attempt;

    vh_libcrypto_path[0] = '\0';
    // Module snapshots can fail transiently while another thread loads or unloads a
    // DLL. Microsoft documents ERROR_BAD_LENGTH as retryable for TH32CS_SNAPMODULE.
    for (attempt = 0; attempt < 10; attempt++) {
        snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
        if (snapshot != INVALID_HANDLE_VALUE || GetLastError() != ERROR_BAD_LENGTH)
            break;
    }
    if (snapshot == INVALID_HANDLE_VALUE)
        return NULL;

    memset(&entry, 0, sizeof(entry));
    entry.dwSize = (DWORD)sizeof(entry);
    if (Module32FirstW(snapshot, &entry)) {
        do {
            if (vh_libcrypto_is_libcrypto_w(entry.szModule)) {
                handle = vh_libcrypto_validate(vh_libcrypto_retain(entry.hModule));
                if (handle != NULL)
                    break;
            }
        } while (Module32NextW(snapshot, &entry));
    }
    CloseHandle(snapshot);

    if (handle != NULL)
        vh_libcrypto_record_module_path(handle);
    return handle;
}

#endif  // _WIN32

// Last resort, when neither enumeration nor the interpreter's own symbol scope produced
// a usable library. On POSIX this can dlopen() by well-known soname, so it may map a
// second copy of OpenSSL. On Windows GetModuleHandleExW() only finds modules that are
// already loaded; this is a second chance by name if the Toolhelp snapshot failed.
static vh_lib_handle vh_libcrypto_open_known(void) {
#if defined(_WIN32)
    static const char *const names[] = {
        "libcrypto-3-x64.dll",
        "libcrypto-3-arm64.dll",
        "libcrypto-3-x86.dll",
        "libcrypto-3.dll",
        "libcrypto-1_1-x64.dll",
        "libcrypto-1_1.dll",
    };
#elif defined(__APPLE__)
    static const char *const names[] = {
        "libcrypto.3.dylib",
        "libcrypto.1.1.dylib",
    };
#else
    static const char *const names[] = {
        "libcrypto.so.3",
        "libcrypto.so.1.1",
    };
#endif
    const size_t nnames = sizeof(names) / sizeof(names[0]);
    size_t i;

    for (i = 0; i < nnames; i++) {
        vh_lib_handle handle;
#if !defined(_WIN32)
        if (vh_libcrypto_set_path(names[i]) != 0)
            return NULL;
#endif
        handle = vh_libcrypto_validate(vh_libcrypto_open(names[i]));
        if (handle == NULL)
            continue;
#if defined(_WIN32)
        vh_libcrypto_record_module_path(handle);
#endif
        return handle;
    }
    return NULL;
}

#if defined(_WIN32)
static void *vh_libcrypto_symbol(vh_lib_handle handle, const char *name) {
    return (void *)(uintptr_t)GetProcAddress(handle, name);
}
#else
static void *vh_libcrypto_symbol(vh_lib_handle handle, const char *name) {
    return dlsym(handle, name);
}
#endif

static void vh_libcrypto_release(vh_lib_handle handle) {
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

// A file named libcrypto need not provide the OpenSSL 1.1+ API we use. In particular,
// OpenSSL 1.0 has EVP_sha256 but lacks EVP_MD_CTX_new/free. Check the complete required
// interface before selecting a candidate so an older copy cannot hide a usable one.
static int vh_libcrypto_has_evp(vh_lib_handle handle) {
    static const char *const names[] = {
        "EVP_sha256",
        "EVP_MD_CTX_new",
        "EVP_MD_CTX_free",
        "EVP_DigestInit_ex",
        "EVP_DigestUpdate",
        "EVP_DigestFinal_ex",
    };
    const size_t nnames = sizeof(names) / sizeof(names[0]);
    size_t i;

    for (i = 0; i < nnames; i++) {
        if (vh_libcrypto_symbol(handle, names[i]) == NULL)
            return 0;
    }
    return 1;
}

// Drop our reference to an incompatible candidate. Successfully selected handles are
// deliberately retained for the process lifetime.
static vh_lib_handle vh_libcrypto_validate(vh_lib_handle handle) {
    if (handle == NULL)
        return NULL;
    if (vh_libcrypto_has_evp(handle))
        return handle;
    vh_libcrypto_release(handle);
    return NULL;
}

// Release a selected handle unless it is the interpreter's own global symbol scope,
// which is not owned by us. Successfully selected handles are retained for the process
// lifetime; this is only used when initialisation fails after the handle was acquired.
static void vh_libcrypto_release_selected(vh_lib_handle handle) {
#if defined(_WIN32)
    vh_libcrypto_release(handle);
#else
    if (handle != RTLD_DEFAULT)
        vh_libcrypto_release(handle);
#endif
}

// Clear the resolved entry points after a failed initialisation. Without this, a
// partially bound EVP interface remains visible to a later retry in a subinterpreter.
static void vh_libcrypto_clear_symbols(void) {
    vh_EVP_MD_CTX_new = NULL;
    vh_EVP_MD_CTX_free = NULL;
    vh_EVP_sha256 = NULL;
    vh_EVP_DigestInit_ex = NULL;
    vh_EVP_DigestUpdate = NULL;
    vh_EVP_DigestFinal_ex = NULL;
    vh_OpenSSL_version = NULL;
}

#ifndef _WIN32
// Whether a statically linked OpenSSL is visible in the interpreter's own symbol scope.
// CPython is linked with --export-dynamic, so such an OpenSSL exports its EVP entry
// points to RTLD_DEFAULT and no library has to be mapped to use it. RTLD_DEFAULT is
// NULL on glibc and musl, so this is a probe rather than a handle test.
static int vh_libcrypto_default_scope_has_evp(void) {
    return vh_libcrypto_has_evp(RTLD_DEFAULT);
}
#endif

// Overwrite the recorded path with the real one, derived from where the bound symbols
// actually live. Until here the recorded name is whatever the loader was handed: a bare
// soname (which resolves against the current working directory) or the ".."-laden path
// h5py's RPATH produces. On Windows the module handle is the answer. If neither works
// the loader name is left in place.
static int vh_libcrypto_set_path_from_bound(vh_lib_handle handle, void *symbol) {
#if defined(_WIN32)
    (void)symbol;
    vh_libcrypto_record_module_path(handle);
    return 0;
#elif defined(VH_HAVE_DLADDR)
    Dl_info info;

    (void)handle;
    if (symbol == NULL)
        return 0;
    memset(&info, 0, sizeof(info));
    if (dladdr(symbol, &info) == 0 || info.dli_fname == NULL
        || info.dli_fname[0] == '\0')
        return 0;
    return vh_libcrypto_set_path(info.dli_fname);
#else
    (void)handle;
    (void)symbol;
    return 0;
#endif
}

// Resolve one symbol into its typed function pointer slot. C does not allow converting
// between object and function pointers, so go through memcpy: both are pointer-sized on
// every ABI we support.
#define VH_LIBCRYPTO_BIND(fn, name)                                 \
    do {                                                            \
        symbol = vh_libcrypto_symbol(handle, name);                 \
        if (symbol == NULL) {                                       \
            missing = name;                                         \
            goto missing_symbol;                                    \
        }                                                           \
        memcpy(&(fn), &symbol, sizeof(void *));                     \
    } while (0)

static int vh_libcrypto_init(void) {
    vh_lib_handle handle;
    int resolved = 0;
    const char *missing = NULL;
    void *symbol;

    if (vh_libcrypto_state != 0)
        return vh_libcrypto_state == 1 ? 0 : -1;

    vh_libcrypto_path[0] = '\0';
    vh_libcrypto_version[0] = '\0';
    vh_libcrypto_error[0] = '\0';

    handle = vh_libcrypto_find_loaded();
    if (handle != NULL)
        resolved = 1;
#if !defined(_WIN32)
    else if (vh_libcrypto_default_scope_has_evp()) {
        handle = RTLD_DEFAULT;
        resolved = 1;
    }
#endif
    if (!resolved) {
        handle = vh_libcrypto_open_known();
        resolved = handle != NULL;
    }
    if (!resolved) {
#if defined(_WIN32)
        vh_libcrypto_fail(
            "no libcrypto DLL is loaded in this process and none of the well-known "
            "names could be opened");
#else
        vh_libcrypto_fail(
            "no libcrypto shared library is loaded in this process, the interpreter's "
            "own symbol scope does not export the EVP functions, and none of the "
            "well-known names could be opened");
#endif
        vh_libcrypto_state = -1;
        return -1;
    }

    // Bind EVP_sha256 first and derive the library path from it right away, so that a
    // missing symbol in any of the remaining binds is reported with the real library
    // name rather than "<unknown libcrypto>".
    VH_LIBCRYPTO_BIND(vh_EVP_sha256, "EVP_sha256");

    // `symbol` now holds one of the entry points we are about to call, so it lives in the
    // library those calls will land in: the authoritative answer to "which library is
    // this?" for LIBCRYPTO_PATH and for the error messages below. A statically linked
    // OpenSSL resolved from RTLD_DEFAULT reports the interpreter executable, which is
    // where those functions really are.
    if (vh_libcrypto_set_path_from_bound(handle, symbol) != 0) {
        vh_libcrypto_release_selected(handle);
        vh_libcrypto_clear_symbols();
        vh_libcrypto_state = -1;
        return -1;
    }

    VH_LIBCRYPTO_BIND(vh_EVP_MD_CTX_new, "EVP_MD_CTX_new");
    VH_LIBCRYPTO_BIND(vh_EVP_MD_CTX_free, "EVP_MD_CTX_free");
    VH_LIBCRYPTO_BIND(vh_EVP_DigestInit_ex, "EVP_DigestInit_ex");
    VH_LIBCRYPTO_BIND(vh_EVP_DigestUpdate, "EVP_DigestUpdate");
    VH_LIBCRYPTO_BIND(vh_EVP_DigestFinal_ex, "EVP_DigestFinal_ex");

    // Optional: only used for the informational version string. OPENSSL_VERSION == 0.
    symbol = vh_libcrypto_symbol(handle, "OpenSSL_version");
    if (symbol != NULL) {
        const char *text;
        memcpy(&vh_OpenSSL_version, &symbol, sizeof(void *));
        text = vh_OpenSSL_version(0);
        if (text != NULL)
            snprintf(vh_libcrypto_version, sizeof(vh_libcrypto_version), "%s", text);
    }

    // Candidate probing can recover from a transient error, such as an overlong path
    // on an earlier candidate. Do not expose that stale error after succeeding.
    vh_libcrypto_error[0] = '\0';
    vh_libcrypto_state = 1;
    return 0;

missing_symbol:
    vh_libcrypto_fail(
        "symbol %s not found in %s",
        missing,
        vh_libcrypto_path[0] != '\0' ? vh_libcrypto_path : "<unknown libcrypto>");
    vh_libcrypto_release_selected(handle);
    vh_libcrypto_clear_symbols();
    vh_libcrypto_state = -1;
    return -1;
}

// Path of the libcrypto the EVP functions were resolved from, as the loader reports it:
// absolute for anything the loader resolved, but it may still contain symlinks (hash.pyx
// applies realpath()). For an OpenSSL linked statically into the interpreter it is the
// path of the interpreter executable.
static const char *vh_libcrypto_get_path(void) {
    return vh_libcrypto_path;
}

// OpenSSL_version(OPENSSL_VERSION) of that library, e.g. "OpenSSL 3.5.1 1 Jul 2025";
// empty if the symbol or the call was unavailable.
static const char *vh_libcrypto_get_version(void) {
    return vh_libcrypto_version;
}

// Human-readable reason for the last vh_libcrypto_init() failure; empty otherwise.
static const char *vh_libcrypto_get_error(void) {
    return vh_libcrypto_error;
}
