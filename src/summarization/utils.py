import re
from spacy.lang.en.stop_words import STOP_WORDS


def entity_match(ent, source, level=2):
    if level == 0:
        # case sensitive match
        if ent in source:
            return [ent,]
        else:
            return []
    elif level == 1:
        # case insensitive match
        if re.search(re.escape(ent), source, re.IGNORECASE):
            return [ent,]
        else:
            return []
    elif level == 2:
        # split entity and match non-stop words
        ent_split = ent.split()
        result = []
        for l in range(len(ent_split), 1, -1):
            for start_i in range(len(ent_split) - l + 1):
                sub_ent = " ".join(ent_split[start_i:start_i+l])
                if re.search(re.escape(sub_ent), source, re.IGNORECASE):
                    result.append(sub_ent)
            if result:
                break
        if result:
            return result
        else:
            for token in ent_split:
                if token.lower() not in STOP_WORDS or token == "US":
                    if re.search(re.escape(token), source, re.IGNORECASE):
                        result.append(token)
            return result
    return []
