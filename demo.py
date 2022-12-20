from gpt_index import GPTTreeIndex, SimpleDirectoryReader, WikipediaReader

topic = "spiderman"

wiki_docs = WikipediaReader().load_data(pages=[topic])

# documents = SimpleDirectoryReader("data").load_data()
index = GPTTreeIndex(wiki_docs)
index.save_to_disk(f'index_{topic}.json')

# index = GPTTreeIndex.load_from_disk(f'index_{topic}.json')
response = index.query(
    "Who is spiderman?", verbose=True
)
