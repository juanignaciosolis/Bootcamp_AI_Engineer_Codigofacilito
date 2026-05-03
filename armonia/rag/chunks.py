from ingestion import chunk_by_sections, load_directory

documents = load_directory("../data/docs")

chunks = chunk_by_sections(*documents)

for chunk in chunks:
    print("\n------- INICIO DEL CHUNK ------")
    print("METADATA: " ,chunk.metadata["tipo"])
    print("\n")
    print(chunk.content)
    print("------- FIN DEL CHUNK ------\n")