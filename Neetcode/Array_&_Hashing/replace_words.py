from typing import List

def replaceWords(dictionary: List[str], sentence: str) -> str:
    root = set(dictionary)
    # break the string sentence in to words and store them in a temp
    words = sentence.split(" ")
    result = []
    # create prefix substring of increasing length.
    for word in words:
        for i in range(len(word) + 1):
            prefix = word[:i]
            # cut only word matching root set
            if prefix in root:
                result.append(prefix)
                break
        else:
            result.append(word)
    return ' '.join(result)


dictionary1 = ["cat","bat","rat"]
sentence1 = "the cattle was rattled by the battery"

print(replaceWords(dictionary1, sentence1))
