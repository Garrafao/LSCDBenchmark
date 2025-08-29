

def tokenize_truncate_decode(sentence, positions, tokenizer, max_seq_len=128):
    
    def center_sentence(input_ids, positions, max_seq_len):
        left = input_ids[:positions[0]]
        right = input_ids[positions[1]:]

        overflow_left = len(left) - int((max_seq_len - len(input_ids[positions[0]:positions[1]])) / 2)

        overflow_right = len(right) - int((max_seq_len - len(input_ids[positions[0]:positions[1]])) / 2)


        if overflow_left > 0 and overflow_right > 0:
            left = left[overflow_left:]
            right = right[:len(right)-overflow_right]
        elif overflow_left > 0 and overflow_right <= 0:
            left = left[overflow_left:]
        else:
            right = right[:len(right)-overflow_right]

        return left + input_ids[positions[0]:positions[1]] + right

    def tokenize_sentence(sentence, positions):
        left, target, right = sentence[:positions[0]], sentence[positions[0]:positions[1]], sentence[positions[1]:]

        token_positions = [0, 0]
        tokens = []

        if left:
            tokens += tokenizer.tokenize(left)
        token_positions[0] = len(tokens)
        tokens += tokenizer.tokenize('<t>')
        target_subtokens = tokenizer.tokenize(target)
        tokens += target_subtokens
        tokens += tokenizer.tokenize('</t>', max_length=128)
        token_positions[1] = len(tokens)
        if right:
            tokens += tokenizer.tokenize(right)

        return tokens, token_positions

    tokens, token_positions = tokenize_sentence(sentence, positions)
    
    n_extra_tokens = 2  
    len_input = len(tokens) + n_extra_tokens
    
    if len_input > max_seq_len:
        tokens = center_sentence(tokens, token_positions, max_seq_len - n_extra_tokens)
        
    
    decoded_text = tokenizer.convert_tokens_to_string(tokens)
    return decoded_text