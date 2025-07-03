import csv
from collections.abc import AsyncGenerator
from typing import IO

from .page import Page
from .parser import Parser

class CSVParser(Parser):
    """
    Concrete parser that can parse CSV into Page objects. Each row becomes a Page object.
    """

    async def parse(self,content : IO) -> AsyncGenerator[Page,None]:
        #Check if content is in bytes and decode to string
        content_str : str
        if isinstance(content,(bytes,bytearray)):
            content_str = content.decode('utf-8')
        elif hasattr(content,'read'):
            content_str = content.read().decode('utf-8')


        #Creating a CSV reader from the text content
        reader = csv.reader(content_str.splitlines())
        offset = 0

        next(reader,None)

        for i,row in enumerate(reader):
            page_text = ','.join(row)
            yield Page(i,offset,page_text)
            offset += len(page_text)  + 1
