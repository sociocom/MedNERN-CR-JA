import re


def split_sentences(text):
    """Given a string, split it into sentences.

    :param text: The string to be processed.
    :return: The list of split sentences.
    """
    processed_text = re.split(
        "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s\n*|(?<=[^A-zＡ-ｚ0-9０-９ ].)(?<=[。．.?？!！])(?![\.」])\n*", text)
    # processed_text = re.split("(? <=[。?？!！])")  # In case only a simple regex is necessary
    processed_text = [x.strip() for x in processed_text]
    processed_text = [x for x in processed_text if x != '']
    return processed_text
