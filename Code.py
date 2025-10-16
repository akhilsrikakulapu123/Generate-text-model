def generate_text(seed_text, next_words=20):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == np.argmax(predicted):
                predicted_word = word
                break
        seed_text += " " + predicted_word
    return seed_text

print(generate_text("Artificial intelligence"))
