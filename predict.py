# -*-coding:UTF-8 -*-
import torch
import pickle
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW
import torch.nn.functional as F     # 激励函数都在这

def toBertIds(question_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question_input)))

if __name__ == "__main__":

    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    pkl_file = open('trained_model/data_features.pkl', 'rb')
    data_features = pickle.load(pkl_file)
    pkl_file.close()
    answer_dic = data_features['answer_dic']
    
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained('trained_model/config.json')
    model = bert_class.from_pretrained('trained_model/pytorch_model.bin', from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()

    with open('valid_data.txt','r',encoding='utf-8') as f:
        data = f.read()
    qa_pairs = data.split("\n")

    questions = []
    answers = []
    for qa_pair in qa_pairs:
        qa_pair = qa_pair.split()
        try:
            a,q = qa_pair
            questions.append(q)
            answers.append(a)
        except:
            continue

    count = 0
    for index, question in enumerate(questions):
        bert_ids = toBertIds(question)
        assert len(bert_ids) <= 512
        input_ids = torch.LongTensor(bert_ids).unsqueeze(0)

        # predict時，因為沒有label所以沒有loss
        outputs = model(input_ids)

        prediction = torch.max(F.softmax(outputs[0]), dim = 1)[1] # 在第1維度取最大值並返回索引值 
        predict_label = prediction.data.cpu().numpy().squeeze() 

        ans_label = answer_dic.to_text(predict_label)
        if ans_label == answers[index]:
            count += 1

    print("總共" + str(len(questions)) + "題，正確率為:" + str((count/len(questions))*100) + "%")