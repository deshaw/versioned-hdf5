/*
 * SHA-256 — vendored from B-Con's public-domain crypto-algorithms
 * (https://github.com/B-Con/crypto-algorithms).  No modifications beyond
 * adding this header comment and using standard C99 integer types.
 */
#ifndef VH5_SHA256_H
#define VH5_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define SHA256_DIGEST_SIZE 32

typedef struct {
    uint8_t  data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} SHA256_CTX;

void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const void *data, size_t len);
void sha256_final(SHA256_CTX *ctx, uint8_t hash[SHA256_DIGEST_SIZE]);

#endif /* VH5_SHA256_H */
