/* bpe.h — GPT-2 byte-level BPE tokenizer (tiktoken "gpt2" compatible), header-only.
 *
 * Ports tiktoken's RAW-BYTE BPE: tokens are byte strings with contiguous ids 0..n-1.
 * Encode = gpt2 regex pretokenize, then for each piece repeatedly merge the adjacent
 * byte-pair whose concatenation has the LOWEST id, until none mergeable. Decode = the
 * raw bytes of each token concatenated.
 *
 * Regex pretokenizer replicates tiktoken's gpt2 pattern:
 *   '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
 * ASCII-EXACT. Bytes >= 0x80 are treated as \p{L} (Latin-script text is exact; non-Latin
 * digits/punctuation are approximated). Documented limitation.
 *
 * No model dependencies — shared by run.c and runq.c.
 */
#ifndef BPE_H
#define BPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

typedef struct {
    int n;              // number of byte-tokens (50256 for gpt2)
    int eot;            // <|endoftext|> id
    unsigned char* blob;// all token bytes concatenated
    int* tok_off;       // [n] offset into blob
    int* tok_len;       // [n] length
    int* ht;            // open-addressing hash table -> id (or -1)
    int ht_cap;         // power of two
} GPT2Tokenizer;

/* --- FNV-1a over a byte range --- */
static unsigned int bpe_hash(const unsigned char* p, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static int bpe_streq(const GPT2Tokenizer* t, int id, const unsigned char* p, int len) {
    if (t->tok_len[id] != len) return 0;
    return memcmp(t->blob + t->tok_off[id], p, len) == 0;
}

/* return id of byte range [p,p+len) or -1 if not a token */
static int gpt2_lookup(const GPT2Tokenizer* t, const unsigned char* p, int len) {
    unsigned int mask = (unsigned int)(t->ht_cap - 1);
    unsigned int slot = bpe_hash(p, len) & mask;
    while (t->ht[slot] != -1) {
        if (bpe_streq(t, t->ht[slot], p, len)) return t->ht[slot];
        slot = (slot + 1) & mask;
    }
    return -1;
}

static void gpt2_ht_insert(GPT2Tokenizer* t, int id) {
    unsigned int mask = (unsigned int)(t->ht_cap - 1);
    const unsigned char* p = t->blob + t->tok_off[id];
    unsigned int slot = bpe_hash(p, t->tok_len[id]) & mask;
    while (t->ht[slot] != -1) slot = (slot + 1) & mask;
    t->ht[slot] = id;
}

static GPT2Tokenizer gpt2_load(const char* path) {
    GPT2Tokenizer t; memset(&t, 0, sizeof(t));
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "couldn't open gpt2 tokenizer %s\n", path); exit(EXIT_FAILURE); }
    if (fread(&t.n, sizeof(int), 1, f) != 1 || fread(&t.eot, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "bad gpt2 tokenizer header\n"); exit(EXIT_FAILURE);
    }
    if (t.n <= 0 || t.n > 10000000 || t.eot < 0) {
        fprintf(stderr, "gpt2 tokenizer: implausible n=%d eot=%d (corrupt file?)\n", t.n, t.eot);
        exit(EXIT_FAILURE);
    }
    t.tok_off = (int*)malloc(t.n * sizeof(int));
    t.tok_len = (int*)malloc(t.n * sizeof(int));
    // read records, building blob
    int blob_cap = 1 << 20, blob_len = 0;
    t.blob = (unsigned char*)malloc(blob_cap);
    for (int id = 0; id < t.n; id++) {
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) { fprintf(stderr, "bad token %d\n", id); exit(EXIT_FAILURE); }
        if (len < 0 || len > (1 << 20)) { fprintf(stderr, "bad token length %d for id %d\n", len, id); exit(EXIT_FAILURE); }
        while (blob_len + len > blob_cap) { blob_cap *= 2; t.blob = (unsigned char*)realloc(t.blob, blob_cap); }
        if (len > 0 && fread(t.blob + blob_len, 1, len, f) != (size_t)len) { fprintf(stderr, "bad token bytes %d\n", id); exit(EXIT_FAILURE); }
        t.tok_off[id] = blob_len; t.tok_len[id] = len; blob_len += len;
    }
    fclose(f);
    // hash table: power of two >= 2n
    t.ht_cap = 1; while (t.ht_cap < t.n * 2) t.ht_cap <<= 1;
    t.ht = (int*)malloc(t.ht_cap * sizeof(int));
    for (int i = 0; i < t.ht_cap; i++) t.ht[i] = -1;
    for (int id = 0; id < t.n; id++) gpt2_ht_insert(&t, id);
    return t;
}

static void gpt2_free(GPT2Tokenizer* t) {
    free(t->blob); free(t->tok_off); free(t->tok_len); free(t->ht);
}

/* decode: bytes of a token id (len via out param). eot / out-of-range -> empty. */
static const unsigned char* gpt2_token_bytes(const GPT2Tokenizer* t, int id, int* len) {
    if (id < 0 || id >= t->n) { *len = 0; return (const unsigned char*)""; }
    *len = t->tok_len[id];
    return t->blob + t->tok_off[id];
}

/* --- ASCII-exact char classes (byte>=0x80 treated as letter) --- */
static int bpe_is_space(unsigned char c) {
    return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\v'||c=='\f';
}
static int bpe_is_letter(unsigned char c) {
    return (c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>=0x80);
}
static int bpe_is_digit(unsigned char c) { return c>='0'&&c<='9'; }
static int bpe_is_other(unsigned char c) { /* non-space, non-letter, non-digit */
    return !bpe_is_space(c) && !bpe_is_letter(c) && !bpe_is_digit(c);
}

/* Return the END (exclusive) of the next pretoken piece starting at buf[p], p<len. */
static int gpt2_next_piece(const unsigned char* buf, int len, int p) {
    unsigned char c = buf[p];
    /* 1) '(?:[sdmt]|ll|ve|re) */
    if (c == '\'' && p + 1 < len) {
        unsigned char d = buf[p+1];
        if (d=='s'||d=='d'||d=='m'||d=='t') return p+2;
        if (p + 2 < len) {
            unsigned char e = buf[p+2];
            if ((d=='l'&&e=='l')||(d=='v'&&e=='e')||(d=='r'&&e=='e')) return p+3;
        }
        /* else fall through to other rules */
    }
    /* 2-4)  ?\p{L}++ |  ?\p{N}++ |  ?[^\s\p{L}\p{N}]++   (optional single leading space) */
    {
        int sp = (c == ' ') ? 1 : 0;
        int r = p + sp;
        if (r < len) {
            unsigned char d = buf[r];
            if (bpe_is_letter(d)) { while (r < len && bpe_is_letter(buf[r])) r++; return r; }
            if (bpe_is_digit(d))  { while (r < len && bpe_is_digit(buf[r]))  r++; return r; }
            if (bpe_is_other(d))  { while (r < len && bpe_is_other(buf[r]))  r++; return r; }
        }
    }
    /* whitespace handling (c must be whitespace here) */
    if (bpe_is_space(c)) {
        int e = p; while (e < len && bpe_is_space(buf[e])) e++;  // maximal ws run [p,e)
        int L = e - p;
        /* 5) \s++$  : run reaches end of string */
        if (e == len) return e;
        /* 6) \s+(?!\S) : leave the last space for the following word (needs L>=2) */
        if (L >= 2) return p + (L - 1);
        /* 7) \s : single whitespace */
        return p + 1;
    }
    /* safety fallback: consume one byte (should not happen) */
    return p + 1;
}

/* byte-pair merge of piece [s,e); appends token ids to out, advancing *n_out. */
static void gpt2_bpe_piece(const GPT2Tokenizer* t, const unsigned char* buf,
                           int s, int e, int* off, int* slen, int* out, int* n_out) {
    int count = e - s;
    if (count <= 0) return;
    for (int i = 0; i < count; i++) { off[i] = s + i; slen[i] = 1; }
    while (count >= 2) {
        int best = -1, best_rank = INT_MAX;
        for (int i = 0; i < count - 1; i++) {
            int id = gpt2_lookup(t, buf + off[i], slen[i] + slen[i+1]);  // contiguous concat
            if (id >= 0 && id < best_rank) { best_rank = id; best = i; }
        }
        if (best < 0) break;
        slen[best] += slen[best+1];
        for (int j = best+1; j < count-1; j++) { off[j] = off[j+1]; slen[j] = slen[j+1]; }
        count--;
    }
    for (int i = 0; i < count; i++) {
        int id = gpt2_lookup(t, buf + off[i], slen[i]);   // single bytes always present
        out[(*n_out)++] = id;
    }
}

/* encode text -> token ids. caller provides out[] with capacity >= strlen(text). */
static void gpt2_encode(const GPT2Tokenizer* t, const char* text, int* out, int* n_out) {
    const unsigned char* buf = (const unsigned char*)text;
    int len = (int)strlen(text);
    *n_out = 0;
    if (len == 0) return;
    int* off = (int*)malloc(len * sizeof(int));
    int* slen = (int*)malloc(len * sizeof(int));
    int p = 0;
    while (p < len) {
        int q = gpt2_next_piece(buf, len, p);
        if (q <= p) q = p + 1;  // guard
        gpt2_bpe_piece(t, buf, p, q, off, slen, out, n_out);
        p = q;
    }
    free(off); free(slen);
}

#endif /* BPE_H */
