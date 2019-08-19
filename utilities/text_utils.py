import re
from utilities.constants import STOP_POS, PUNKT, CODE_SNIPPET_MIN_CHARS, CODE_SNIPPET_MIN_TOKENS
from nltk.tag import pos_tag
from nltk import PorterStemmer
from itertools import chain
from operator import itemgetter
import json
from collections import Counter
import pandas as pd

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def load_SO_tags(tag_csv_filename, filter_by_count=-1, remove_textless=True, synonym_filename=None):
    tags_df = pd.read_csv(tag_csv_filename)
    tags_df['TagName'] = tags_df['TagName'].astype(str)
    tags_df['Excerpt'] = tags_df['Excerpt'].astype(str)
    tags_df['TagCount'] = tags_df['TagCount'].astype(int)
    if filter_by_count > 0:
        tags_df = tags_df.loc[tags_df.TagCount > filter_by_count]
    if remove_textless:
        tags_df = tags_df.loc[tags_df.Excerpt != 'nan']
    tags = set(tags_df.TagName.tolist())
    # we transform each tag with a - into two versions: one where it's replaced by a space, and one where it's
    # replaced by '' (i.e. nothing).
    tag_to_original_mapping = dict()

    if synonym_filename is not None:
        synonyms = json.load(open(synonym_filename, 'r', encoding='utf8'))
        # We remove from our target tags those tags that are considered to be synonyms of a master tag.
        tags = {tag for tag in tags if tag not in synonyms}
        # The if at the end is necessary if we want to avoid putting back in tags that we have removed due to being
        # descriptionless or under the count threshold.
        tag_to_original_mapping.update({x.replace('-', ' '): synonyms[x] for x in synonyms if synonyms[x] in tags})
        tag_to_original_mapping.update({x.replace('-', ''): synonyms[x] for x in synonyms if synonyms[x] in tags})

    tag_to_original_mapping.update({x.replace('-', ' '): x for x in tags})
    tag_to_original_mapping.update({x.replace('-', ''): x for x in tags})

    tag_n_gram = dict(Counter([len(x.split('-')) for x in tags]))
    print(tag_n_gram)

    return tag_to_original_mapping, tags, tags_df

def get_tags(tags_list):
    opening_str = '&lt;'
    closing_str = '&gt;'
    opening_tag = [m.end() for m in re.finditer(opening_str, tags_list)]
    closing_tag = [m.start() for m in re.finditer(closing_str, tags_list)]

    result_list = []
    for i in range(len(opening_tag)):
        result_list.append(tags_list[opening_tag[i]:closing_tag[i]])

    return result_list

def clean_html(raw_html):
  cleanr = re.compile('<.*?>')
  clean_text = re.sub(cleanr, ' ', raw_html)
  return clean_text

def generate_n_grams(tokens, return_string=True, n=3):
    if n == 1 or tokens == []:
        return tokens
    if len(tokens) >= n:
        n_gram_list = [tokens[i:i+n] for i in range(len(tokens)-n+1)]
        if return_string:
            n_gram_list = ['_'.join(x) for x in n_gram_list]
        return n_gram_list + generate_n_grams(tokens, return_string=return_string, n=n-1)
    else:
        return generate_n_grams(tokens, return_string=return_string, n=n-1)

def remove_all_code(text):
    opening_str = '&lt;code&gt;'
    closing_str = '&lt;/code&gt;'
    opening_code = [(m.start(), 1) for m in re.finditer(opening_str, text)]
    if len(opening_code) == 0:
        return text
    closing_code = [(m.start(), -1) for m in re.finditer(closing_str, text)]

    all_code = sorted(opening_code+closing_code, key=itemgetter(0))

    code_tag_counter = 0
    text_counter = 0
    result = ''

    result += text[text_counter:all_code[code_tag_counter][0]]

    while code_tag_counter < len(all_code):
        depth_counter = all_code[code_tag_counter][1]
        starting_code_tag_counter = code_tag_counter
        while depth_counter > 0:
            code_tag_counter += 1
            try:
                depth_counter += all_code[code_tag_counter][1]
            except Exception as e:
                raise Exception(text + '\n' + str(depth_counter) + '\n' + str(code_tag_counter))
        code_between_tags = text[all_code[starting_code_tag_counter][0]+len(opening_str):all_code[code_tag_counter][0]]
        if len(code_between_tags.split()) < CODE_SNIPPET_MIN_TOKENS and len(code_between_tags) < CODE_SNIPPET_MIN_CHARS:
            result += code_between_tags
        text_counter = all_code[code_tag_counter][0] + len(closing_str)
        if code_tag_counter == len(all_code) - 1:
            end_point = len(text)
        else:
            end_point = all_code[code_tag_counter+1][0]
        code_tag_counter += 1
        result += text[text_counter:end_point]

    return result

def tokenise_stem_punkt_and_stopword(text, punkt_to_remove=PUNKT, remove_periods=False, remove_numbers=False
                                     , stopword_set=None, ngram_stopwords=None, do_stem = False,
                                     remove_code=False):

    """
    Handles text and tokenises it. By default lowercases, removes all HTML tags and escaped characters such as
    &nbsp;, etc. Has options for removing punctuation, numbers, stopwords and also for stemming.
    """

    if (text is None):
        return []

    # Lowercases the whole text.
    text = text.lower()

    if (remove_code):
        text = remove_all_code(text)
    # Removes HTML tags that are one character long. The whitespace is necessary to distinguish between an innocent <
    # (which is ' &lt; ') and a < that's part of an HTML tag. Same with next regex and &gt;.
    # This comes first because the second pattern would not understand these and would skip to the next &gt;.
    text = re.sub(r'(<(\S){1}>)', ' ', text)
    # Removes HTML tags more than 1 character long.
    text = re.sub(r'(<(\S){1}(.*?)(\S){1}>)', ' ', text)
    # Removes any remaining special, escaped characters.
    text = re.sub(r'&[^&;]+;', ' ', text)
    # N-gram stopwords are removed before tokenisation. This feature should be used cautiously, avoid using it for
    # unigrams and be careful that the ngrams couldn't be a substring of a bigger n-gram.
    if ngram_stopwords is not None:
        for ngram in ngram_stopwords:
            text = text.replace(ngram, '')
    # Tokenises and removes punctuation.
    if punkt_to_remove is not None:
        tokenised = " ".join("".join([" " if ch in punkt_to_remove else ch for ch in text]).split()).split()
    else:
        if remove_periods:
            text = re.sub(r'\.+ ', ' ', text)
        tokenised = text.split()

    # Removes stopwords and numbers.
    if remove_numbers:
        tokenised = [x for x in tokenised if not is_number(x)]
    if stopword_set is not None:
        tokenised = [x for x in tokenised if x not in stopword_set]
    if do_stem:
        stemmer = PorterStemmer()
        return [stemmer.stem(x) for x in tokenised]
    else:
        return tokenised

def get_ngram_bags_for_tags(text, n_grams=4, stop_POS=STOP_POS, stopwords=None):
    """
    Takes a piece of text, generates ngrams, removes undesired parts of speech and returns a bag of words.
    :param text: The piece of text
    :param n_grams: number of ngrams to generate
    :param stop_POS: Parts of Speech to remove
    :return: A bag of ngrams
    """

    # We first do POS tagging, then generate n-grams, then delete the unigrams that are in stop_POS. Stopwords
    # are not removed since we wanna match these with tags later, so it makes no sense to remove things
    # that could end up being polysemous. We could then remove number unigrams and such, but those would not
    # match with any tags, so we can simply leave them be -- and that's what we'll do.

    # For a full list of POS tags, see here:
    # https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b

    custom_punkt = ''.join([x for x in PUNKT if x not in ['.', '-']])
    tokenised_with_pos = pos_tag(tokenise_stem_punkt_and_stopword(text, punkt_to_remove=custom_punkt,
                                                                  remove_numbers=False, stopword_set=None,
                                                                  ngram_stopwords=None,
                                                                  do_stem=False))
    ngrams_with_pos = generate_n_grams(tokenised_with_pos, return_string=False, n=n_grams)
    n_grams_stringified = []
    for ngram in ngrams_with_pos:
        if isinstance(ngram, list) and len(ngram) > 1:
            n_grams_stringified.append(' '.join([y[0] for y in ngram]))
        else:
            if ngram[1] not in stop_POS:
                n_grams_stringified.append(ngram[0])
    return n_grams_stringified

def clean_from_forbidden_tags(tag_list, forbidden_mapping):
    """
    Takes a list of tags and the mapping from tags to their unrelated subtags, and cleans the list from the unrelated
    subtags. Bear in mind that this will remove those subtags from the tag_list regardless of whether they were observed
    as a subtag of the larger tag, or independently, but since the mapping is between unrelated tags, this should not
    pose a problem. Nevertheless, do not use for very large tag sets.
    :param tag_list: List of tags to clean.
    :param forbidden_mapping: Mapping from each tag to its subtags that are irrelevant to it.
    :return: Cleaned list of tags
    """

    forbidden_list = set(chain.from_iterable(forbidden_mapping[x] for x in tag_list if x in forbidden_mapping))
    #print([x for x in tag_list if x in forbidden_list])
    return [x for x in tag_list if x not in forbidden_list]

def find_matching_tags(n_gram_list, tags_to_originals):
    list_of_originals = []
    for n_gram in n_gram_list:
        current_original_tag = tags_to_originals.get(n_gram, None)
        if current_original_tag is not None:
            list_of_originals.append(current_original_tag)
    return list_of_originals

def create_entity_item_bags(entity_text, so_tags, forbidden_subtag_map,
                              n_grams=4, textual_field_to_use='summary'):
    new_col_name = textual_field_to_use + '_tags'
    entity_text[new_col_name] = entity_text[textual_field_to_use]. \
        apply(lambda x: get_ngram_bags_for_tags(clean_html(x), n_grams))

    # This is where the tags get detected and the forbidden tags are eliminated.
    entity_text[new_col_name] = entity_text[new_col_name].apply(lambda x:
                                                                clean_from_forbidden_tags(
                                                                find_matching_tags(x, so_tags),
                                                                forbidden_subtag_map))
