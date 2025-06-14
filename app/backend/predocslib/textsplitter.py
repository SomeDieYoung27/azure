import logging
from abc import ABC
from collections.abc import Generator

import tiktoken

from .page import Page,SplitPage

logger = logging.getLogger(__name__)


class TextSpiltter(ABC):
    """
    Splits a list of pages into smaller chunks
    :param pages: The pages to split
    :return: A generator of SplitPage
    """

    def split_pages(self,pages : list[Page]) -> Generator[SplitPage,None,None]:
        if False:
            yield


ENCODING_MODEL = "text-embedding-ada-002"

STANDARD_WORD_BREAKS = [",", ";", ":", " ", "(", ")", "[", "]", "{", "}", "\t", "\n"]

CJK_WORD_BREAKS = [
    "、",
    "，",
    "；",
    "：",
    "（",
    "）",
    "【",
    "】",
    "「",
    "」",
    "『",
    "』",
    "〔",
    "〕",
    "〈",
    "〉",
    "《",
    "》",
    "〖",
    "〗",
    "〘",
    "〙",
    "〚",
    "〛",
    "〝",
    "〞",
    "〟",
    "〰",
    "–",
    "—",
    "‘",
    "’",
    "‚",
    "‛",
    "“",
    "”",
    "„",
    "‟",
    "‹",
    "›",
]

STANDARD_SENTENCE_ENDINGS = [".", "!", "?"]

# See CL05 and CL06, based on JIS X 4051:2004
# https://www.w3.org/TR/jlreq/#cl-04
CJK_SENTENCE_ENDINGS = ["。", "！", "？", "‼", "⁇", "⁈", "⁉"]

bpe = tiktoken.encoding_for_model(ENCODING_MODEL)

DEFAULT_OVERLAP_PERCENT = 10  # See semantic search article for 10% overlap performance
DEFAULT_SECTION_LENGTH = 1000  # Roughly 400-500 tokens for English


class SentenceTextSpiltter(TextSpiltter):
    """
    Class that splits pages into smaller chunks. This is required because embedding models may not be able to analyze an entire page at once
    """

    def __init__(self,max_tokens_per_section : int = 500):
        self.sentence_endings = STANDARD_SENTENCE_ENDINGS + CJK_SENTENCE_ENDINGS
        self.word_breaks = STANDARD_WORD_BREAKS + CJK_WORD_BREAKS
        self.max_section_length = DEFAULT_SECTION_LENGTH
        self.sentence_search_limit = 100
        self.max_tokens_per_section = max_tokens_per_section
        self.section_overlap = int(self.max_section_length * DEFAULT_OVERLAP_PERCENT / 100)
        
        
    def split_page_by_max_tokens(self,page_num : int,text : str) -> Generator[SplitPage,None,None]:
        """
        Recursively you split the page by maximum number of tokens to find better languages with higher token/word
        limits
        """
        tokens = bpe.encode(text)
        if len(tokens) <= self.max_tokens_per_section:
            #Section is already within max tokens,return it
            yield SplitPage(page_num=page_num,text=text)

        else:
            start = int(len(text) // 2)
            pos = 0
            boundary = int(len(text) // 2)
            split_position = -1

            while start - pos > boundary:
                if text[start-pos] in self.sentence_endings:
                    split_position = start - pos
                    break

                elif text[start + pos] in self.sentence_endings:
                    split_position = start + pos
                    break

                else:
                    pos += 1


            if split_position > 0:
                first_half = text[: split_position + 1]
                second_half = text[split_position + 1 :]

            else:
                middle = int(len(text) // 2)
                overlap = int(len(text) * (DEFAULT_OVERLAP_PERCENT / 100))
                first_half = text[: middle + overlap]
                second_half = text[middle - overlap :]

            yield from self.split_page_by_max_tokens(page_num, first_half)
            yield from self.split_page_by_max_tokens(page_num, second_half)



    def split_pages(self,pages : list[Page]) -> Generator[SplitPage,None,None]:
        def find_page(offset):
            num_pages = len(pages)

            for i in range(num_pages-1):
                if offset >= pages[i].offset and offset < pages[i+1].offset:
                    return pages[i].page_num
                

        all_text = "".join(page.text for page in pages)
        if len(all_text.strip()) == 0:
            return
        
        length = len(all_text)
        if length <= self.max_section_length:
            yield from self.split_page_by_max_tokens(page_num=find_page(0),text=all_text)
            return
        

        start = 0
        end = length

        while start + self.section_overlap < length:
            last_word = -1
            end = start + self.max_section_length

            if end > length:
                end = length

            else:

                while(
                    end < length
                    and (end - start - self.max_section_length) < self.sentence_search_limit
                    and all_text[end] not in self.sentence_endings
                ) :
                    if all_text[end] in self.word_breaks:
                        last_word = end

                    end += 1

                if end < length and all_text[end] not in self.sentence_endings and last_word > 0:
                    end = last_word

                if end < length and all_text[end] not in self.sentence_endings and last_word > 0:
                    end = last_word


                if end < length:
                    end += 1


            last_word = -1

            while(
                start > 0
                and start > end - self.max_section_length - 2 * self.sentence_search_limit
                and all_text[start] not in self.sentence_endings
            ):
                if all_text[start] in self.word_breaks:
                    last_word = start

                start -= 1


            if all_text[start] not in self.sentence_endings and last_word > 0:
                start = last_word
            
            if start > 0:
                start += 1


            section_text = all_text[start:end]
            yield from self.split_page_by_max_tokens(page_num=find_page(start), text=section_text)

            last_figure_start = section_text.rfind("<figure")
            if last_figure_start > 2 * self.sentence_search_limit and last_figure_start > section_text.rfind(
                "</figure"
            ):
                start = min(end - self.section_overlap, start + last_figure_start)
                logger.info(
                    f"Section ends with unclosed figure, starting next section with the figure at page {find_page(start)} offset {start} figure start {last_figure_start}"
                )
            else:
                start = end - self.section_overlap


        if start + self.section_overlap < end:
            yield from  self.split_page_by_max_tokens(page_num=find_page(start), text=all_text[start:end])



class SimpleTextSplitter(TextSpiltter):

    def __init__(self,max_object_length : int = 1000):
        self.max_object_length = max_object_length

    def split_pages(self,pages : list[Page]) -> Generator[SplitPage,None,None]:
        all_text = "".join(page.text for page in pages)

        if len(all_text.strip()) == 0:
            return
        
        length = len(all_text)
        if length <= self.max_object_length:
            yield SplitPage(page_num=0,text=all_text)
            return
        
        for i in range(0,length,self.max_object_length):
             yield SplitPage(page_num=i // self.max_object_length, text=all_text[i : i + self.max_object_length])
        
        return