"""仅合并映射的共用 reducer；清空和删除语义由原 reducer 保留。"""


def merge_records(previous, update):
    return {**(previous or {}), **(update or {})}
