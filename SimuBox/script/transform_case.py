import re
from pathlib import Path

is_first_word = True


def should_capitalize(word):
    global is_first_word
    # 定义不需要大写的介词和副词
    excluded_words = {
        "of",
        "and",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "to",
        "from",
        "as",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "a",
        "an",
        "the",
        "placeholder",
        "via",
    }
    # 如果是第一个单词，或者单词不在排除列表中，返回 True
    if is_first_word or word.lower() not in excluded_words:
        if word.lower() == "placeholder":
            return False
        is_first_word = False
        return True
    else:
        return False


def capitalize_title(title):
    # 使用回调函数，对 title 字段中的每个单词的首字母大写
    global is_first_word
    matches = re.findall(r"{(.*?)}", title)
    temp_title = re.sub(r"{(.*?)}", r"##placeholder##", title)
    processed_title = re.sub(
        r"\b\w+\b",
        lambda match: (
            match.group().capitalize()
            if should_capitalize(match.group())
            else match.group()
        ),
        temp_title,
    )
    for match in matches:
        # print(match)
        processed_title = processed_title.replace(
            "##placeholder##", "{" + match + "}", 1
        )
    is_first_word = True
    return processed_title


def process_bib_file(file_path, save_path):
    with open(file_path, "r", encoding="utf-8") as bib_file:
        bib_data = bib_file.read()

    # 使用正则表达式匹配 title 字段，并应用 capitalize_title 函数
    bib_data = re.sub(
        r"title\s*=\s*{(.*)}",
        lambda match: "title = {" + capitalize_title(match.group(1)) + "}",
        bib_data,
    )

    # 将处理后的数据写回原文件
    with open(save_path, "w", encoding="utf-8") as bib_file:
        bib_file.write(bib_data)


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    source_bib_file_path = current_dir.parent / "main.bib"
    target_bib_file_path = current_dir / "main.bib"
    process_bib_file(source_bib_file_path, target_bib_file_path)
