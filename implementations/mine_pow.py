#!/usr/bin/env python3
"""
Minerador em Python para o exercicio 3.

O enunciado usa uma versao didatica do cabecalho: version, previous_block,
merkle_root, timestamp e nonce, todos serializados em big endian. O hash de
prova de trabalho tambem e SHA-256 simples, nao hash duplo.

Para a busca ser viavel, o miolo da compressao SHA-256 e compilado com numba.
Ainda assim, a implementacao fica em Python e mantem visivel a logica
criptografica usada na atividade.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads


ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS_DIR = ROOT / "solutions"
EX2_OUTPUT = SOLUTIONS_DIR / "exercise02.txt"
EX3_OUTPUT = SOLUTIONS_DIR / "exercise03.txt"
EX2_TXIDS_PATH = ROOT / "data" / "ex02_txid_list.txt"

VERSION = 2
PREVIOUS_BLOCK = "00000000d1145790a8694403d4063f323d499e655c83426834d4ce2f8dd4a2ee"
TIMESTAMP = 1_230_999_306  # Jan 03 2009 16:15:06 UTC.

TARGET_HIGH = np.uint32(0x00000000)
TARGET_NEXT = np.uint32(0xFFFF0000)

INITIAL_STATE = np.array(
    [
        0x6A09E667,
        0xBB67AE85,
        0x3C6EF372,
        0xA54FF53A,
        0x510E527F,
        0x9B05688C,
        0x1F83D9AB,
        0x5BE0CD19,
    ],
    dtype=np.uint32,
)

K = np.array(
    [
        0x428A2F98,
        0x71374491,
        0xB5C0FBCF,
        0xE9B5DBA5,
        0x3956C25B,
        0x59F111F1,
        0x923F82A4,
        0xAB1C5ED5,
        0xD807AA98,
        0x12835B01,
        0x243185BE,
        0x550C7DC3,
        0x72BE5D74,
        0x80DEB1FE,
        0x9BDC06A7,
        0xC19BF174,
        0xE49B69C1,
        0xEFBE4786,
        0x0FC19DC6,
        0x240CA1CC,
        0x2DE92C6F,
        0x4A7484AA,
        0x5CB0A9DC,
        0x76F988DA,
        0x983E5152,
        0xA831C66D,
        0xB00327C8,
        0xBF597FC7,
        0xC6E00BF3,
        0xD5A79147,
        0x06CA6351,
        0x14292967,
        0x27B70A85,
        0x2E1B2138,
        0x4D2C6DFC,
        0x53380D13,
        0x650A7354,
        0x766A0ABB,
        0x81C2C92E,
        0x92722C85,
        0xA2BFE8A1,
        0xA81A664B,
        0xC24B8B70,
        0xC76C51A3,
        0xD192E819,
        0xD6990624,
        0xF40E3585,
        0x106AA070,
        0x19A4C116,
        0x1E376C08,
        0x2748774C,
        0x34B0BCB5,
        0x391C0CB3,
        0x4ED8AA4A,
        0x5B9CCA4F,
        0x682E6FF3,
        0x748F82EE,
        0x78A5636F,
        0x84C87814,
        0x8CC70208,
        0x90BEFFFA,
        0xA4506CEB,
        0xBEF9A3F7,
        0xC67178F2,
    ],
    dtype=np.uint32,
)


@njit(cache=True)
def rotr(value: np.uint32, shift: int) -> np.uint32:
    return np.uint32((value >> shift) | (value << (32 - shift)))


@njit(cache=True)
def compress_block(state: np.ndarray, words_in: np.ndarray) -> np.ndarray:
    words = np.zeros(64, dtype=np.uint32)
    for i in range(16):
        words[i] = words_in[i]

    for i in range(16, 64):
        s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3)
        s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10)
        words[i] = np.uint32(words[i - 16] + s0 + words[i - 7] + s1)

    a, b, c, d, e, f, g, h = state

    for i in range(64):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        choice = np.uint32((e & f) ^ ((~e) & g))
        temp1 = np.uint32(h + s1 + choice + K[i] + words[i])
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        majority = np.uint32((a & b) ^ (a & c) ^ (b & c))
        temp2 = np.uint32(s0 + majority)

        h = g
        g = f
        f = e
        e = np.uint32(d + temp1)
        d = c
        c = b
        b = a
        a = np.uint32(temp1 + temp2)

    out = np.empty(8, dtype=np.uint32)
    out[0] = np.uint32(state[0] + a)
    out[1] = np.uint32(state[1] + b)
    out[2] = np.uint32(state[2] + c)
    out[3] = np.uint32(state[3] + d)
    out[4] = np.uint32(state[4] + e)
    out[5] = np.uint32(state[5] + f)
    out[6] = np.uint32(state[6] + g)
    out[7] = np.uint32(state[7] + h)
    return out


@njit(parallel=True, cache=True)
def search_range(
    mid_state: np.ndarray,
    word0: np.uint32,
    word1: np.uint32,
    start_nonce: np.uint64,
    count: np.uint64,
    workers: int,
) -> tuple[np.uint64, np.ndarray]:
    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
    found_nonces = np.empty(workers, dtype=np.uint64)
    found_hashes = np.zeros((workers, 8), dtype=np.uint32)
    for i in range(workers):
        found_nonces[i] = sentinel

    for worker in prange(workers):
        words = np.zeros(64, dtype=np.uint32)
        words[0] = word0
        words[1] = word1
        words[4] = np.uint32(0x80000000)
        words[15] = np.uint32(640)  # 80 bytes * 8 bits.

        nonce = start_nonce + np.uint64(worker)
        end = start_nonce + count

        while nonce < end:
            words[2] = np.uint32(nonce >> np.uint64(32))
            words[3] = np.uint32(nonce & np.uint64(0xFFFFFFFF))

            for i in range(16, 64):
                s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^ (words[i - 15] >> 3)
                s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^ (words[i - 2] >> 10)
                words[i] = np.uint32(words[i - 16] + s0 + words[i - 7] + s1)

            a = mid_state[0]
            b = mid_state[1]
            c = mid_state[2]
            d = mid_state[3]
            e = mid_state[4]
            f = mid_state[5]
            g = mid_state[6]
            h = mid_state[7]

            for i in range(64):
                s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
                choice = np.uint32((e & f) ^ ((~e) & g))
                temp1 = np.uint32(h + s1 + choice + K[i] + words[i])
                s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
                majority = np.uint32((a & b) ^ (a & c) ^ (b & c))
                temp2 = np.uint32(s0 + majority)

                h = g
                g = f
                f = e
                e = np.uint32(d + temp1)
                d = c
                c = b
                b = a
                a = np.uint32(temp1 + temp2)

            h0 = np.uint32(mid_state[0] + a)
            h1 = np.uint32(mid_state[1] + b)
            if h0 == TARGET_HIGH and h1 <= TARGET_NEXT:
                found_nonces[worker] = nonce
                found_hashes[worker, 0] = h0
                found_hashes[worker, 1] = h1
                found_hashes[worker, 2] = np.uint32(mid_state[2] + c)
                found_hashes[worker, 3] = np.uint32(mid_state[3] + d)
                found_hashes[worker, 4] = np.uint32(mid_state[4] + e)
                found_hashes[worker, 5] = np.uint32(mid_state[5] + f)
                found_hashes[worker, 6] = np.uint32(mid_state[6] + g)
                found_hashes[worker, 7] = np.uint32(mid_state[7] + h)
                break

            nonce += np.uint64(workers)

    best_nonce = sentinel
    best_hash = np.zeros(8, dtype=np.uint32)
    for worker in range(workers):
        if found_nonces[worker] < best_nonce:
            best_nonce = found_nonces[worker]
            for i in range(8):
                best_hash[i] = found_hashes[worker, i]

    return best_nonce, best_hash


def read_merkle_root() -> str:
    if EX2_OUTPUT.exists():
        for line in EX2_OUTPUT.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lower()
            if stripped:
                return stripped

    txids = [
        line.strip().lower()
        for line in EX2_TXIDS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    level = [bytes.fromhex(txid) for txid in txids]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]

    return level[0].hex()


def words_from_block(block: bytes) -> np.ndarray:
    return np.array(
        [int.from_bytes(block[i : i + 4], "big") for i in range(0, len(block), 4)],
        dtype=np.uint32,
    )


def build_header_prefix(merkle_root: str) -> bytes:
    return (
        VERSION.to_bytes(4, "big")
        + bytes.fromhex(PREVIOUS_BLOCK)
        + bytes.fromhex(merkle_root)
        + TIMESTAMP.to_bytes(4, "big")
    )


def header_from_nonce(prefix: bytes, nonce: int) -> bytes:
    return prefix + nonce.to_bytes(8, "big")


def digest_words_to_hex(words: np.ndarray) -> str:
    return "".join(f"{int(word):08x}" for word in words)


def self_test(prefix: bytes, mid_state: np.ndarray) -> None:
    second_block = bytearray(64)
    second_block[0:8] = prefix[64:72]
    second_block[16] = 0x80
    second_block[62:64] = (640).to_bytes(2, "big")

    digest_words = compress_block(mid_state, words_from_block(second_block))
    expected = hashlib.sha256(header_from_nonce(prefix, 0)).hexdigest()
    actual = digest_words_to_hex(digest_words)
    if actual != expected:
        raise RuntimeError(f"falha no autoteste SHA-256: {actual} != {expected}")


def prepare_mining_state() -> tuple[str, bytes, np.ndarray, np.uint32, np.uint32]:
    merkle_root = read_merkle_root()
    prefix = build_header_prefix(merkle_root)
    first_block = prefix[:64]
    second_seed = prefix[64:72]

    mid_state = compress_block(INITIAL_STATE, words_from_block(first_block))
    self_test(prefix, mid_state)

    word0 = np.uint32(int.from_bytes(second_seed[:4], "big"))
    word1 = np.uint32(int.from_bytes(second_seed[4:], "big"))
    return merkle_root, prefix, mid_state, word0, word1


def mine(
    workers: int,
    batch_size: int,
    start_nonce: int,
    max_batches: int | None = None,
) -> tuple[bytes, str, int, int, float]:
    merkle_root, prefix, mid_state, word0, word1 = prepare_mining_state()
    count = (batch_size // workers) * workers
    if count <= 0:
        raise ValueError("batch_size precisa ser pelo menos igual ao numero de workers")

    set_num_threads(workers)
    print(f"merkle root: {merkle_root}")
    print(f"minerando com {workers} threads Python/JIT")

    # Compila antes de iniciar a contagem de tempo da busca.
    search_range(mid_state, word0, word1, np.uint64(0), np.uint64(workers), workers)

    started = time.perf_counter()
    attempts = 0
    nonce = start_nonce
    batches = 0

    while nonce < 2**64:
        found_nonce, found_hash = search_range(
            mid_state,
            word0,
            word1,
            np.uint64(nonce),
            np.uint64(count),
            workers,
        )
        attempts += count

        if found_nonce != np.uint64(0xFFFFFFFFFFFFFFFF):
            header = header_from_nonce(prefix, int(found_nonce))
            elapsed = time.perf_counter() - started
            return header, digest_words_to_hex(found_hash), int(found_nonce), attempts, elapsed

        elapsed = time.perf_counter() - started
        rate = attempts / max(elapsed, 1e-9)
        print(f"tentativas={attempts:,} taxa={rate:,.0f}/s")
        nonce += count
        batches += 1

        if max_batches is not None and batches >= max_batches:
            raise RuntimeError("limite de batches atingido sem encontrar nonce valido")

    raise RuntimeError("nenhum nonce valido encontrado no espaco de 64 bits")


def benchmark(workers: int, batch_size: int) -> None:
    merkle_root, _prefix, mid_state, word0, word1 = prepare_mining_state()
    set_num_threads(workers)
    count = (batch_size // workers) * workers
    print(f"merkle root: {merkle_root}")
    print(f"autoteste SHA-256 passou; compilando busca com {workers} threads")

    search_range(mid_state, word0, word1, np.uint64(0), np.uint64(workers), workers)
    started = time.perf_counter()
    search_range(mid_state, word0, word1, np.uint64(0), np.uint64(count), workers)
    elapsed = time.perf_counter() - started
    print(f"benchmark: {count:,} hashes em {elapsed:.2f}s ({count / elapsed:,.0f}/s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=80_000_000)
    parser.add_argument("--start-nonce", type=int, default=0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        benchmark(workers=args.workers, batch_size=args.batch_size)
        return

    header, block_hash, nonce, attempts, elapsed = mine(
        workers=args.workers,
        batch_size=args.batch_size,
        start_nonce=args.start_nonce,
        max_batches=args.max_batches,
    )

    EX3_OUTPUT.write_text(header.hex() + "\n", encoding="utf-8")
    print(f"header: {header.hex()}")
    print(f"hash:   {block_hash}")
    print(f"nonce:  {nonce}")
    print(f"tries:  {attempts:,}")
    print(f"time:   {elapsed:.2f}s")


if __name__ == "__main__":
    main()
