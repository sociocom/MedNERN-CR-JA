"""XML to IOB2 converter.

This script contains functions to convert XML-tagged text to IOB2 format.

Example:

        iobs = xml2iob('私は<C value="N">宇宙人</C>', tag_lists=['C'], attr=['value'], tokenizer=list)

        print_iob(iobs)


@Author: Gabriel Andrade

@Date: 2023-11-13
"""

from collections import deque

import lxml.etree as etree

def __convert_xml_to_taglist(sent, tag_list=None, attr=[], ignore_mismatch_tags=True):
    text = '<sent>' + sent + '</sent>'

    # Adding recover parameter allows handling missing tags.
    # It will reject closing tags with not start.
    # It will consider the start tag to span until the end of the sentence, if not close is found.
    parser = etree.XMLPullParser(['start', 'end'], recover=not ignore_mismatch_tags)
    parser.feed(text)

    ne_type = "O"
    ne_prefix = ""
    res = ""
    label = []
    tag_set = deque()
    s_pos = -1
    idx = 0
    word = ''

    for event, elem in parser.read_events():
        isuse = (tag_list is None or (tag_list is not None and elem.tag in tag_list))

        if event == 'start':
            # assert len(tag_set) < 2, "タグが入れ子になっています\n{}".format(sent)
            s_pos = idx

            if attr is not None and elem.attrib:
                attr_list = '-'.join([v for k, v in elem.attrib.items() if k in attr])
                attr_list = '-' + attr_list if attr_list != '' else ''
            else:
                attr_list = ''

            word = elem.text if elem.text is not None else ""
            res += word
            idx += len(word)

            if elem.tag != 'sent' and isuse:
                label_list = [s_pos, idx, elem.tag + attr_list, word, elem.tag]
                tag_set.append(label_list)
                # label.append((s_pos, idx, elem.tag + attr_list, word))

        if event == 'end':
            if elem.tag != 'sent' and isuse and tag_set[-1][-1] == elem.tag:
                # and tag_set[-1] == elem.tag:
                label_list = tag_set.pop()
                label.append(tuple(label_list[:-1]))
                for tag in tag_set:
                    tag[1] = idx
                    tag[3] += word
            word = elem.tail if elem.tail is not None else ""
            res += word
            idx += len(word)

    return res, label

def __convert_taglist_to_iob(sent, entities, tokenizer=list):
    splitted = []
    position = 0
    for entity in entities:
        start = entity[0]
        end = entity[1]
        label = entity[2]
        splitted.append({'text': sent[position:start], 'label': 'O'})
        splitted.append({'text': sent[start:end], 'label': label})
        position = end
    splitted.append({'text': sent[position:], 'label': 'O'})
    splitted = [s for s in splitted if s['text']]

    tokens = []
    labels = []

    for s in splitted:
        tokens_s = tokenizer(s['text'])
        label_s = s['label']
        if label_s != 'O':
            labels_s = ['B-' + label_s] + ['I-' + label_s] * (len(tokens_s) - 1)
        else:
            labels_s = ['O'] * len(tokens_s)

        tokens.extend(tokens_s)
        labels.extend(labels_s)

    return list(zip(tokens, labels))
def xml2iob(sent, tag_list=None, attr_list=None, tokenizer=list, ignore_mismatch_tags=True):
    """Convert XML to IOB2 format.

     Some tags and attributes can be ignored by setting tag_list and attr parameters.

    Args:
        sent (str): Input string in XML-tag format.
        tag_list (List): List of tags to be converted to IOB. If None, ALL tags will be converted. Default: None.
        attr_list (List): List of attributes to be converted to IOB. If None, attributes will NOT be added to the IOB tags. Default: None.
        tokenizer (callable): Tokenize function used to split the sentence into a list of tokens. Default: Each character is a token.
        ignore_mismatch_tags (bool): Should it try to recover if tags are mismatched?. If false, it will raise an exception on any error. Default: True.

    Returns:
        List (tuple): List of (token, IOB2 tag).
    """
    res, label = __convert_xml_to_taglist(sent, tag_list=tag_list, attr=attr_list, ignore_mismatch_tags=ignore_mismatch_tags)
    iob = __convert_taglist_to_iob(res, label, tokenizer=tokenizer)
    return [item for item in iob if item[0] != '\n']

# Example usafe
if __name__ == '__main__':
    # Default usage
    print('Default usage')
    text = 'This is a <c key="test">test</c> <a>string containing multiple</a> <c val="test2">tags</c>.'
    tags = xml2iob(text)
    print(tags, '\n')

    # Using BERT tokenizer
    print('Using BERT tokenizer')
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tags = xml2iob(text, tokenizer=tokenizer.tokenize)
    print(tags, '\n')

    # Using custom tokenizer
    print('Using custom tokenizer')
    def my_tokenizer(text):
        return text.split(' ')
    tags = xml2iob(text, tokenizer=my_tokenizer)
    print(tags, '\n')

    # Filtering tags
    print('Filtering tags')
    tags = xml2iob(text, tag_list=['c'], attr_list=['key'], tokenizer=my_tokenizer)
    print(tags, '\n')
