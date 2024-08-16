def tokens_to_indices(tokens, vocab):
    return [vocab["<bos>"]] + [vocab[token] for token in tokens] + [vocab["<eos>"]]


def generate_batch(source_tokenizer, target_tokenizer):
    src_vocab = source_tokenizer.get_vocab()
    trg_vocab = target_tokenizer.get_vocab()

    def f(data_batch):
        src_data = []
        trg_data = []

        for src, trg in data_batch:
            src_data.append(
                torch.tensor(
                    tokens_to_indices(source_tokenizer.tokenize(src), src_vocab)
                )
            )
            trg_data.append(
                torch.tensor(
                    tokens_to_indices(target_tokenizer.tokenize(trg), trg_vocab)
                )
            )

        src_data = pad_sequence(
            src_data, padding_value=src_vocab["<pad>"], batch_first=True
        )
        trg_data = pad_sequence(
            trg_data, padding_value=trg_vocab["<pad>"], batch_first=True
        )

        return src_data, trg_data

    return f


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def init_decoder_cross_attention_weights(source_model, target_model):
    for src_block, trg_block in zip(
        source_model.decoder.blocks, target_model.decoder.blocks
    ):
        trg_block.cross_attn.load_state_dict(src_block.cross_attn.state_dict())
        trg_block.cross_attn_layer_norm.load_state_dict(
            src_block.cross_attn_layer_norm.state_dict()
        )


def init_decoder_self_attention_weights(source_model, target_model):
    for src_block, trg_block in zip(
        source_model.decoder.blocks, target_model.decoder.blocks
    ):
        trg_block.self_attn.load_state_dict(src_block.self_attn.state_dict())
        trg_block.self_attn_layer_norm.load_state_dict(
            src_block.self_attn_layer_norm.state_dict()
        )


# def init_pfn_weights(source_model, target_model):
#     for src_block, trg_block in zip(source_model.decoder.blocks, target_model.decoder.blocks):
#         trg_block.pfn.load_state_dict(src_block.pfn.state_dict())
#         trg_block.pfn_layer_norm.load_state_dict(src_block.pfn_layer_norm.state_dict())


def init_encoder_weights(source_model, target_model):
    target_model.encoder.load_state_dict(source_model.encoder.state_dict())
