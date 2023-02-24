from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import faiss
from typing import List
from langchain.docstore.document import Document
from unstructured.cleaners.core import clean_bullets
from unstructured.documents.elements import (
    Address,
    Element,
    ElementMetadata,
    ListItem,
    NarrativeText,
    Text,
    Title,
)
import re 
from unstructured.partition.text_type import (
    is_possible_narrative_text,
    is_possible_title,
    is_bulleted_text,
    is_us_city_state_zip,
)

file_path = "toolsqa.txt"

PARAGRAPH_PATTERN = "\n\n\n|\n\n|\r\n|\r|\n"  # noqa: W605 NOTE(harrell)
def split_by_paragraph(content: str) -> List[str]:
    return re.split(PARAGRAPH_PATTERN, content)


with open(file_path, "r") as f:
    file_text = f.read()
file_content = split_by_paragraph(file_text)

elements: List[Element] = list()
metadata = ElementMetadata(filename=file_path)
for ctext in file_content:
    ctext = ctext.strip()

    if ctext == "":
        continue
    if is_bulleted_text(ctext):
        elements.append(ListItem(text=clean_bullets(ctext), metadata=metadata))
    elif is_us_city_state_zip(ctext):
        elements.append(Address(text=ctext, metadata=metadata))
    elif is_possible_narrative_text(ctext):
        elements.append(NarrativeText(text=ctext, metadata=metadata))
    elif is_possible_title(ctext):
        elements.append(Title(text=ctext, metadata=metadata))
    else:
        elements.append(Text(text=ctext, metadata=metadata))

metadata = {"source": file_path}
text = "\n\n".join([str(el) for el in elements])
docs = [Document(page_content=text, metadata=metadata)]

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Load Data to vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
faiss.write_index(vectorstore.index, "vectorstore.index")

# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)