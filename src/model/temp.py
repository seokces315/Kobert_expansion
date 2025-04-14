def x():
    # Tokenize function
    def recursive_tokenize(text):
        encoded_text = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            stride=128,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )
        return encoded_text

    # Splitting, Tokenizing text
    csat_kor_dataset["text"] = csat_kor_dataset["text"].map(recursive_tokenize)

    # Generate chunk-wise embeddings
    print()
    print("< Chunk-wise Embedding >")
    print()

    brk = 0
    dataset_length = len(csat_kor_dataset["text"])
    model.eval()
    for text_dict in csat_kor_dataset["text"]:
        # Logging
        if brk % 100 == 0:
            print(f"[Current/Total Corpus] : {brk}/{dataset_length}")
            print()

        # Model forwarding
        input_dict = {
            k: v.to(device)
            for k, v in text_dict.items()
            if k in ["input_ids", "attention_mask"]
        }
        with torch.no_grad():
            output = model(**input_dict)
            if brk % 100 == 0:
                print(output.last_hidden_state.shape)
                print()

        brk += 1
    print(f"[Current/Total Corpus] : {brk}/{dataset_length}")
    print()
