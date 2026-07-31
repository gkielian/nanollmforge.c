/* bpe_test.c — read lines from stdin, print "id id id ..." per line using bpe.h.
 * Build: gcc -O2 -o bpe_test bpe_test.c
 * Use:   ./bpe_test tokenizer_gpt2.bin < lines.txt
 * Validated against tiktoken in Python.
 */
#include "bpe.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s tokenizer_gpt2.bin < lines\n", argv[0]); return 1; }
    GPT2Tokenizer t = gpt2_load(argv[1]);
    char* line = NULL; size_t cap = 0; ssize_t n;
    int* ids = NULL; size_t ids_cap = 0;
    while ((n = getline(&line, &cap, stdin)) != -1) {
        if (n > 0 && line[n-1] == '\n') { line[n-1] = '\0'; n--; }  // strip newline
        if ((size_t)n + 1 > ids_cap) { ids_cap = n + 1; ids = (int*)realloc(ids, ids_cap * sizeof(int)); }
        int nid = 0;
        gpt2_encode(&t, line, ids, &nid);
        for (int i = 0; i < nid; i++) printf(i ? " %d" : "%d", ids[i]);
        printf("\n");
    }
    free(line); free(ids); gpt2_free(&t);
    return 0;
}
