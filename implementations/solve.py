#!/usr/bin/env python3
"""
Gera as respostas dos exercicios 1 e 2.

A atividade simplifica o Bitcoin: todos os hashes sao SHA-256 simples e os
bytes sao tratados em big endian. O codigo abaixo segue essa convencao para
evitar a confusao comum com a serializacao real dos blocos Bitcoin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOLUTIONS_DIR = ROOT / "solutions"

MEMPOOL_PATH = DATA_DIR / "mempool.csv"
EX2_TXIDS_PATH = DATA_DIR / "ex02_txid_list.txt"
EX1_OUTPUT = SOLUTIONS_DIR / "exercise01.txt"
EX2_OUTPUT = SOLUTIONS_DIR / "exercise02.txt"

WEIGHT_LIMIT = 4_000_000
REQUIRED_BLOCK_TXID = (
    "4c50e3dad7f98bceb6441f96b23748dea84fbdb7cedd603441e6ea4a574d04a6"
)
REQUIRED_PROOF_TXID = (
    "49ff8cccf1ca12179e9ae7a4760f550b5a18401b27e1e057604e27c3e10c08fb"
)


@dataclass(frozen=True)
class Transaction:
    txid: str
    fee: int
    weight: int
    parents: tuple[str, ...]


def sha256(data: bytes) -> bytes:
    """SHA-256 simples, como exigido pelo enunciado."""
    return hashlib.sha256(data).digest()


def read_mempool(path: Path = MEMPOOL_PATH) -> dict[str, Transaction]:
    mempool: dict[str, Transaction] = {}

    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.reader(csv_file):
            txid = row[0].strip().lower()
            parents = ()
            if len(row) >= 4 and row[3].strip():
                parents = tuple(p.strip().lower() for p in row[3].split(";") if p)

            mempool[txid] = Transaction(
                txid=txid,
                fee=int(row[1]),
                weight=int(row[2]),
                parents=parents,
            )

    return mempool


def build_package_orders(
    mempool: dict[str, Transaction],
) -> dict[str, tuple[str, ...]]:
    """
    Para cada tx, pre-calcula o pacote minimo que precisa entrar no bloco.

    O pacote vem em ordem topologica: pais primeiro, filho depois. Assim, ao
    incluir o pacote no bloco, a regra de ancestralidade ja fica satisfeita.
    """

    orders: dict[str, tuple[str, ...]] = {}
    visiting: set[str] = set()

    def visit(txid: str) -> tuple[str, ...]:
        if txid in orders:
            return orders[txid]
        if txid in visiting:
            raise ValueError(f"ciclo de dependencias detectado em {txid}")

        visiting.add(txid)
        ordered: list[str] = []
        seen: set[str] = set()

        for parent in mempool[txid].parents:
            for ancestor in visit(parent):
                if ancestor not in seen:
                    ordered.append(ancestor)
                    seen.add(ancestor)

        if txid not in seen:
            ordered.append(txid)

        visiting.remove(txid)
        orders[txid] = tuple(ordered)
        return orders[txid]

    for txid in mempool:
        visit(txid)

    return orders


def package_stats(
    package: list[str], mempool: dict[str, Transaction]
) -> tuple[int, int]:
    fee = sum(mempool[txid].fee for txid in package)
    weight = sum(mempool[txid].weight for txid in package)
    return fee, weight


def select_transactions(mempool: dict[str, Transaction]) -> list[str]:
    """
    Seleciona transacoes por taxa/peso efetiva de pacote.

    Em Bitcoin, uma transacao com pais nao pode ser avaliada isoladamente se os
    pais ainda nao entraram no bloco. Por isso a heuristica compara o conjunto
    "transacao + ancestrais ausentes" e adiciona o melhor pacote que couber.
    """

    package_orders = build_package_orders(mempool)
    selected: set[str] = set()
    block: list[str] = []
    current_weight = 0

    def add_package(package: list[str]) -> None:
        nonlocal current_weight
        for txid in package:
            if txid in selected:
                continue
            selected.add(txid)
            block.append(txid)
            current_weight += mempool[txid].weight

    # A transacao obrigatoria entra primeiro com todos os ancestrais necessarios.
    required_package = list(package_orders[REQUIRED_BLOCK_TXID])
    add_package(required_package)

    while True:
        remaining_weight = WEIGHT_LIMIT - current_weight
        best_package: list[str] | None = None
        best_fee = -1
        best_weight = 1

        for txid in mempool:
            if txid in selected:
                continue

            package = [
                candidate
                for candidate in package_orders[txid]
                if candidate not in selected
            ]
            fee, weight = package_stats(package, mempool)
            if weight > remaining_weight:
                continue

            # Compara f/w sem ponto flutuante: f1/w1 > f2/w2.
            if (
                best_package is None
                or fee * best_weight > best_fee * weight
                or (fee * best_weight == best_fee * weight and fee > best_fee)
            ):
                best_package = package
                best_fee = fee
                best_weight = weight

        if best_package is None:
            break

        add_package(best_package)

    return block


def read_txids(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def merkle_root_and_proof(
    txids: list[str], target_txid: str
) -> tuple[str, list[str]]:
    """
    Calcula a raiz de Merkle e a prova de inclusao do alvo.

    Cada nivel combina pares como sha256(left || right). Quando sobra um no
    final do nivel, ele e duplicado, exatamente como descrito na atividade.
    """

    if target_txid not in txids:
        raise ValueError(f"txid alvo nao encontrado: {target_txid}")

    target_index = txids.index(target_txid)
    level = [bytes.fromhex(txid) for txid in txids]
    proof: list[str] = []

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])

        sibling_index = target_index ^ 1
        proof.append(level[sibling_index].hex())

        next_level: list[bytes] = []
        for i in range(0, len(level), 2):
            next_level.append(sha256(level[i] + level[i + 1]))

        target_index //= 2
        level = next_level

    return level[0].hex(), proof


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def solve() -> None:
    SOLUTIONS_DIR.mkdir(exist_ok=True)

    mempool = read_mempool()
    selected_txids = select_transactions(mempool)
    write_lines(EX1_OUTPUT, selected_txids)

    txids = read_txids(EX2_TXIDS_PATH)
    merkle_root, proof = merkle_root_and_proof(txids, REQUIRED_PROOF_TXID)
    write_lines(EX2_OUTPUT, [merkle_root, *proof])

    total_fee = sum(mempool[txid].fee for txid in selected_txids)
    total_weight = sum(mempool[txid].weight for txid in selected_txids)
    print(f"exercise01: {len(selected_txids)} txs, {total_fee} sats, {total_weight} weight")
    print(f"exercise02: merkle root {merkle_root}, proof with {len(proof)} siblings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ex1",
        action="store_true",
        help="gera apenas a solucao do exercicio 2",
    )
    args = parser.parse_args()

    if args.skip_ex1:
        txids = read_txids(EX2_TXIDS_PATH)
        merkle_root, proof = merkle_root_and_proof(txids, REQUIRED_PROOF_TXID)
        write_lines(EX2_OUTPUT, [merkle_root, *proof])
        print(f"exercise02: merkle root {merkle_root}, proof with {len(proof)} siblings")
        return

    solve()


if __name__ == "__main__":
    main()
