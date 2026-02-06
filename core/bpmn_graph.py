"""
BPMN Graph Module - Представление и анализ BPMN диаграмм в виде графа.

Модуль для преобразования графа элементов BPMN диаграммы
в текстовое описание алгоритма.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union


def _format_text(s: Optional[str]) -> str:
    """
    Форматирует текст для вывода: убирает \\t и \\n, лишние пробелы,
    оставляет только латиницу, кириллицу, цифры и пробелы.
    """
    if s is None:
        return ""
    s = str(s).replace("\t", " ").replace("\n", " ")
    s = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9 ]", "", s)
    s = re.sub(r" +", " ", s)
    return s.strip()


NodeType = Literal["start", "task", "end", "document", "parallel", "exclusive", "receive_message", "send_message"]


@dataclass
class Actor:
    """
    Актор (действующее лицо) в BPMN диаграмме.

    Attributes:
        id: Уникальный идентификатор актора.
        name: Имя актора (роль).
    """
    id: int
    name: str


@dataclass
class BpmnNode:
    """
    Узел BPMN диаграммы в аналитическом представлении графа.
    
    Attributes:
        id: Уникальный идентификатор ноды (целое число).
        node_type: Тип ноды — start, task, end, document, parallel, exclusive, receive_message, send_message.
        text: Текстовое содержимое ноды (если есть). Для задач — описание,
              для start/end — описание события.
        actor_id: Идентификатор актора (действующего лица), к которому
                  прикреплена нода.
    """
    id: int
    node_type: NodeType
    text: Optional[str]
    actor_id: int

    def __post_init__(self) -> None:
        valid_types = ("start", "task", "end", "document", "parallel", "exclusive", "receive_message", "send_message")
        if self.node_type not in valid_types:
            raise ValueError(
                f"node_type должен быть один из {valid_types}, получено: {self.node_type}"
            )

    def __str__(self) -> str:
        text_part = f": {self.text}" if self.text else ""
        return f"[{self.node_type}] Node #{self.id} (actor={self.actor_id}){text_part}"


@dataclass
class BpmnGraph:
    """
    Граф BPMN диаграммы — совокупность вершин, рёбер и акторов.

    Attributes:
        nodes: Список всех вершин графа (нод).
        edges: Список пар (id1, id2) — стрелка от ноды с id1 к ноде с id2.
        actors: Список акторов (id и имя).
    """
    nodes: List[BpmnNode]
    edges: List[Tuple[int, int]]
    actors: List[Actor]


def parse_bpmn_graph(source: Union[str, Path, dict]) -> BpmnGraph:
    """
    Парсит BpmnGraph из JSON.

    Args:
        source: Путь к JSON-файлу, строка с JSON или уже распарсенный dict.

    Returns:
        BpmnGraph с нодами и рёбрами.

    Raises:
        ValueError: При неверной структуре данных.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(source)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"source должен быть str, Path или dict, получено: {type(source)}")

    if "nodes" not in data:
        raise ValueError("JSON должен содержать поле 'nodes'")
    if "edges" not in data:
        raise ValueError("JSON должен содержать поле 'edges'")

    nodes = []
    for i, node_data in enumerate(data["nodes"]):
        if isinstance(node_data, dict):
            node_id = node_data.get("id")
            if node_id is None:
                raise ValueError(f"Нода #{i}: отсутствует поле 'id'")
            node_type = node_data.get("node_type")
            valid_types = ("start", "task", "end", "document", "parallel", "exclusive", "receive_message", "send_message")
            if node_type not in valid_types:
                raise ValueError(
                    f"Нода #{node_id}: node_type должен быть один из {valid_types}, получено: {node_type}"
                )
            text = node_data.get("text")
            actor_id = node_data.get("actor_id")
            if actor_id is None:
                actor_id = -1  # Специальное значение: актор не задан
            nodes.append(BpmnNode(id=node_id, node_type=node_type, text=text, actor_id=int(actor_id)))
        else:
            raise ValueError(f"Нода #{i}: ожидается объект, получено: {type(node_data)}")

    edges = []
    for i, edge in enumerate(data["edges"]):
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((int(edge[0]), int(edge[1])))
        else:
            raise ValueError(f"Ребро #{i}: ожидается пара [id1, id2], получено: {edge}")

    actors = []
    if "actors" in data:
        actors_data = data["actors"]
        if isinstance(actors_data, list):
            for i, a in enumerate(actors_data):
                if isinstance(a, dict):
                    aid = a.get("id")
                    name = a.get("name")
                    if aid is None or name is None:
                        raise ValueError(f"Актор #{i}: нужны поля 'id' и 'name'")
                    actors.append(Actor(id=int(aid), name=str(name)))
                else:
                    raise ValueError(f"Актор #{i}: ожидается объект, получено: {type(a)}")
        elif isinstance(actors_data, dict):
            for kid, name in actors_data.items():
                actors.append(Actor(id=int(kid), name=str(name)))
        else:
            raise ValueError("'actors' должен быть списком или объектом")

    return BpmnGraph(nodes=nodes, edges=edges, actors=actors)


def _emit(text: str, out: Optional[List[str]] = None) -> None:
    """Выводит текст: в out или через print."""
    if out is not None:
        out.append(text)
    else:
        print(text)


def _process_node(
    nid: int,
    node: BpmnNode,
    nodes_by_id: dict,
    actor_by_id: dict,
    incoming: dict,
    outgoing: dict,
    step_num: str,
    indent: str,
    out: Optional[List[str]] = None,
) -> None:
    """Обрабатывает одну ноду — выводит соответствующий текст с номером шага и отступом."""
    def _actor_name(aid: int) -> str:
        return _format_text(actor_by_id.get(aid, "(Неизвестно)"))

    def _fmt(s: Optional[str], default: str) -> str:
        return _format_text(s) or default

    if node.node_type == "task":
        text = _fmt(node.text, "(без текста)")
        actor_name = _actor_name(node.actor_id)
        line = f"{indent}{step_num} - {text} - {actor_name}"
        doc_preds = [
            nodes_by_id[pid] for pid in incoming[nid]
            if nodes_by_id[pid].node_type == "document"
        ]
        if doc_preds:
            doc_texts = [_fmt(d.text, "(без названия)") for d in doc_preds]
            if len(doc_texts) == 1:
                line += f"\n{indent}   Для задачи используется документ: {doc_texts[0]}"
            else:
                line += f"\n{indent}   Для задачи используются документы: {', '.join(doc_texts)}"
        _emit(line, out)
    elif node.node_type == "document":
        text = _fmt(node.text, "(без названия)")
        actor_name = _actor_name(node.actor_id)
        _emit(f"{indent}{step_num} - {actor_name} создает документ {text}.", out)
        task_preds = [
            nodes_by_id[pid] for pid in incoming[nid]
            if nodes_by_id[pid].node_type == "task"
        ]
        if task_preds:
            source_parts = [
                f"{_fmt(t.text, '(без текста)')} ({_actor_name(t.actor_id)})"
                for t in task_preds
            ]
            _emit(f"{indent}   Источник документа: \"{'; '.join(source_parts)}\"", out)
    elif node.node_type == "parallel":
        pass
    elif node.node_type == "exclusive":
        out_count = len(outgoing.get(nid, []))
        if out_count >= 2:
            actor_name = _actor_name(node.actor_id)
            _emit(f"{indent}{step_num} - {actor_name} принимает решение", out)
    elif node.node_type == "receive_message":
        text = _fmt(node.text, "(без текста)")
        actor_name = _actor_name(node.actor_id)
        _emit(f"{indent}{step_num} - {actor_name} получил сообщение {text}", out)
    elif node.node_type == "send_message":
        text = _fmt(node.text, "(без текста)")
        actor_name = _actor_name(node.actor_id)
        _emit(f"{indent}{step_num} - {actor_name} отправил сообщение {text}", out)
    elif node.node_type == "end":
        text = _fmt(node.text, "завершён")
        _emit(f"{indent}{step_num} - Алгоритм закончен: {text}.", out)


def _parent_context(
    step_prefix: str, nesting: int, main_step: list[int]
) -> tuple[str, list[int], int]:
    """
    Вычисляет родительский контекст при выходе из merge (exclusive с 1 исходящим).
    Поднимаемся ровно на один уровень вложенности.
    Возвращает (parent_prefix, parent_counter, parent_nesting).

    Структура step_prefix: "K" (решение) или "K.N" (ветка N решения K) или "K.N.M" (ветка M
    вложенного решения K.N). При выходе из merge убираем один уровень:
    - "14.5.2" (ветка 2 решения 14.5) -> "14.5", [2], следующий шаг 14.5.3
    - "14.5.2.3" (ветка 3 вложенного решения 14.5.2) -> "14.5", [2], следующий шаг 14.5.3
    - "14.5" (ветка 5 решения 14) -> "14", [5], следующий шаг 14.6
    - "14" (решение 14) -> "", main_step=14, следующий шаг 15
    """
    parts = step_prefix.split(".") if step_prefix else []
    if len(parts) <= 1:
        base = int(parts[0]) if parts else main_step[0]
        main_step[0] = base
        return "", [0], 0
    # Убираем один уровень: ветка K.N -> решение K (убираем 1 сегмент),
    # вложенная ветка K.N.M -> решение K.N (убираем 2 сегмента).
    # len>=4: "14.5.2.3" -> "14.5", [2]; len==3: "14.5.2" -> "14.5", [2]; len==2: "14.5" -> "14", [5]
    to_remove = 2 if len(parts) >= 4 else 1
    parent_prefix = ".".join(parts[:-to_remove])
    parent_counter = [int(parts[-to_remove])]
    parent_nesting = max(0, nesting - 1)
    if not parent_prefix:
        main_step[0] = int(parts[0])
        parent_counter = [0]
    return parent_prefix, parent_counter, parent_nesting


def traverse_and_output(graph: BpmnGraph, out: Optional[List[str]] = None) -> None:
    """
    Проходит по графу в стиле постфиксного обхода (post-order / depth-first).

    Стартовые ноды: тип start ИЛИ любая вершина без входящих рёбер.
    Обходит только компоненту связности, содержащую стартовые ноды.
    Если граф не связен, выводит предупреждение и продолжает по достижимой части.

    При exclusive-разветвлении: вывод «Актор принимает решение» как шаг K,
    затем каждый путь — «Вариант № i:» с отступом, шаги в ветке — K.i.1, K.i.2, ...

    Args:
        graph: BPMN граф для обхода.
        out: Список для сбора вывода. Если None — вывод идёт в print().
    """
    nodes_by_id = {n.id: n for n in graph.nodes}
    actor_by_id = {a.id: a.name for a in graph.actors}

    incoming: dict[int, set[int]] = {n.id: set() for n in graph.nodes}
    outgoing: dict[int, list[int]] = {n.id: [] for n in graph.nodes}
    for src, dst in graph.edges:
        incoming[dst].add(src)
        outgoing[src].append(dst)

    # Стартовые ноды: тип start ИЛИ вершина без входящих рёбер (только исходящие)
    start_ids = sorted({
        n.id for n in graph.nodes
        if n.node_type == "start" or len(incoming[n.id]) == 0
    })
    visited: set[int] = set(start_ids)

    main_step = [0]

    def explore_from(
        nid: int,
        step_prefix: str,
        variant_counter: list[int],
        nesting: int,
    ) -> None:
        """
        step_prefix: '' для основного потока, 'K' или 'K.N' для веток (номер решения, без номера ветки)
        variant_counter: [счётчик] — общий для всех веток под одним решением, даёт уникальные K.1, K.2, K.3...
        nesting: уровень вложенности (N табов для «Вариант №», N+1 для шагов)
        """
        tab = "\t"
        indent = tab * nesting if step_prefix else ""

        out_nodes = sorted(outgoing.get(nid, []))
        for next_nid in out_nodes:
            if next_nid in visited:
                continue
            if not incoming[next_nid].issubset(visited):
                continue
            visited.add(next_nid)
            node = nodes_by_id[next_nid]

            if node.node_type == "exclusive":
                out_count = len(outgoing.get(next_nid, []))
                if out_count == 1:
                    # Выход из выбора — поднимаемся ровно на один уровень вверх
                    _emit(f"{indent}Надо сделать выбор", out)
                    merge_target, = outgoing.get(next_nid, [])
                    parent_prefix, parent_counter, parent_nesting = _parent_context(
                        step_prefix, nesting, main_step
                    )
                    if merge_target not in visited and incoming[merge_target].issubset(visited):
                        visited.add(merge_target)
                        merge_node = nodes_by_id[merge_target]
                        if merge_node.node_type != "parallel":
                            if parent_prefix:
                                parent_counter[0] += 1
                                next_step = f"{parent_prefix}.{parent_counter[0]}"
                            else:
                                main_step[0] += 1
                                next_step = str(main_step[0])
                            parent_indent = tab * parent_nesting if parent_prefix else ""
                            _process_node(
                                merge_target, merge_node, nodes_by_id, actor_by_id,
                                incoming, outgoing, next_step, parent_indent, out,
                            )
                        else:
                            _process_node(
                                merge_target, merge_node, nodes_by_id, actor_by_id,
                                incoming, outgoing, "", "", out,
                            )
                    explore_from(merge_target, parent_prefix, parent_counter, parent_nesting)
                elif out_count >= 2:
                    # Принятие решения — это шаг K
                    if step_prefix:
                        variant_counter[0] += 1
                        step_num = f"{step_prefix}.{variant_counter[0]}"
                    else:
                        main_step[0] += 1
                        step_num = str(main_step[0])
                    _process_node(
                        next_nid, node, nodes_by_id, actor_by_id,
                        incoming, outgoing, step_num, "" if not step_prefix else indent, out,
                    )
                    # У каждого варианта свой счётчик — нумерация с 1: K.1, K.2, ...
                    branches = sorted(outgoing.get(next_nid, []))
                    for i, branch_start in enumerate(branches, 1):
                        variant_indent = tab * nesting
                        _emit(f"{variant_indent}Вариант № {i}:", out)
                        branch_indent = tab * (nesting + 1)
                        visited.add(branch_start)
                        branch_node = nodes_by_id[branch_start]
                        branch_counter = [0]  # Свой счётчик для каждого варианта
                        if branch_node.node_type != "parallel":
                            branch_counter[0] += 1
                            first_step = f"{step_num}.{branch_counter[0]}"
                        else:
                            first_step = ""
                        _process_node(
                            branch_start, branch_node, nodes_by_id, actor_by_id,
                            incoming, outgoing, first_step, branch_indent, out,
                        )
                        explore_from(branch_start, step_num, branch_counter, nesting + 1)
                else:
                    # 0 выходов — просто отмечаем пройденным
                    explore_from(next_nid, step_prefix, variant_counter, nesting)
            else:
                # Обычная нода
                if step_prefix:
                    if node.node_type != "parallel":
                        variant_counter[0] += 1
                        step_num = f"{step_prefix}.{variant_counter[0]}"
                    else:
                        step_num = ""
                else:
                    if node.node_type != "parallel":
                        main_step[0] += 1
                        step_num = str(main_step[0])
                    else:
                        step_num = ""
                _process_node(
                    next_nid, node, nodes_by_id, actor_by_id,
                    incoming, outgoing, step_num, indent, out,
                )
                explore_from(next_nid, step_prefix, variant_counter, nesting)

    for start_id in start_ids:
        start_node = nodes_by_id[start_id]
        # Ноды start и parallel не выводятся — сразу переходим к исходящим
        if start_node.node_type not in ("start", "parallel"):
            main_step[0] += 1
            step_num = str(main_step[0])
            _process_node(
                start_id, start_node, nodes_by_id, actor_by_id,
                incoming, outgoing, step_num, "", out,
            )
        explore_from(start_id, "", [0], 0)

    unvisited = set(nodes_by_id) - visited
    if unvisited:
        _emit(
            "\nВнимание: граф не связен. Некоторые ноды недостижимы из стартовых: "
            f"{sorted(unvisited)}",
            out,
        )
        _emit("Пройдена только компонента связности, содержащая стартовые ноды.", out)



def bpmn_graph_to_text(json_source: Union[str, Path, dict]) -> str:
    """
    Парсит BPMN из JSON и возвращает текстовое описание графа и алгоритма действий.

    Args:
        json_source: Путь к JSON-файлу, строка с JSON или уже распарсенный dict.

    Returns:
        Строка с результатом парсинга и алгоритмом действий.
    """
    graph = parse_bpmn_graph(json_source)
    lines: List[str] = []

    lines.append("\n" + "=" * 50)
    lines.append("РЕЗУЛЬТАТ ПАРСИНГА BPMN ГРАФА:")
    lines.append("=" * 50)

    lines.append("\nАкторы:")
    for actor in graph.actors:
        lines.append(f"  {actor.id}: {actor.name}")

    lines.append("\nНоды:")
    for node in graph.nodes:
        lines.append(f"  {node}")

    lines.append("\nРёбра (стрелки):")
    for src, dst in graph.edges:
        lines.append(f"  {src} -> {dst}")

    lines.append("\n" + "=" * 50)
    lines.append(f"Всего: {len(graph.actors)} акторов, {len(graph.nodes)} нод, {len(graph.edges)} рёбер")
    lines.append("=" * 50)

    lines.append("\n" + "-" * 50)
    lines.append("АЛГОРИТМ ДЕЙСТВИЙ:")
    lines.append("-" * 50)
    traverse_and_output(graph, out=lines)

    return "\n".join(lines)
