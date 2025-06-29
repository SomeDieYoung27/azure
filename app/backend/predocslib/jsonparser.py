import json
from collections.abc import AsyncGenerator
from typing import IO

from .page import Page
from .parser import Parser

class JSONParser(Parser):
    """
    Concrete parser that can parse JSON into Page objects. A top-level object becomes a single Page, while a top-level array becomes multiple Page objects.
    """

    async def parse(self,content : IO) -> AsyncGenerator[Page,None]:
        offset = 0
        data = json.loads(content.read())
        if isinstance(data,list):
            for i,obj in enumerate(data):
                offset += 1
                page_text = json.dumps(obj)
                offset +=len(page_text)

        elif isinstance(data,dict):
            yield Page(0,0,json.dumps(data))