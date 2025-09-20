import pandas

def sanitize(text):
    out = ''
    text = " ".join(text.casefold().split())
    for t in text:
        if 96 < ord(t) < 123 or t in (" ", ","):
            out += t
    return out

def preprocess(row):
    if isinstance(row, pandas.Series):
        combined_text = " ".join([sanitize(row[k]) for k in ('language', 'origin_query', 'category_path')])
    elif isinstance(row, str):
        combined_text = sanitize(row)
    elif isinstance(row, list):
        combined_text = " ".join([sanitize(row[k]) for k in range(3)])
    else:
        raise TypeError("Input of invalid type.")
    return combined_text
