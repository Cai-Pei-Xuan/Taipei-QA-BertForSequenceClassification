# -*-coding:UTF-8 -*-
from core import AnsDic,QuestionDic
from transformers import BertTokenizer
import pickle

# 將答案轉換成數字，為了做分類時可以比較好進行訓練
def make_ans_dic(answers):
    ansdic = AnsDic(answers)
    print("全部答案:",len(ansdic))
    print("全部答案種類:",ansdic.types)
    
    # 測試轉換
    a_id = ansdic.to_id('臺北市信義區公所')
    a_text = ansdic.to_text(a_id)
    assert a_text == '臺北市信義區公所'
    assert ansdic.to_id(a_text) == a_id

    return ansdic

def make_question_dic(quetsions):
    return QuestionDic(quetsions)
    

def convert_data_to_feature(FileName):
    with open(FileName,'r',encoding='utf-8') as f:
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
    
    assert len(answers) == len(questions)
    
    ans_dic = make_ans_dic(answers)
    question_dic = make_question_dic(questions)
    
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')

    q_tokens = []
    max_seq_len = 0
    for q in question_dic.data:
        # print(tokenizer.tokenize(q))
        bert_ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q)))
        if(len(bert_ids)>max_seq_len):
            max_seq_len = len(bert_ids)
        q_tokens.append(bert_ids)
        # print(bert_ids)
        # print(tokenizer.convert_ids_to_tokens(tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q)))))
    
    print("最長問句長度:",max_seq_len)
    assert max_seq_len <= 512 # 小於BERT-base長度限制

    original_length = []
    # 補齊長度
    for q in q_tokens:
        # 紀錄原本長度
        length = len(q)
        original_length.append(length)
        while len(q)<max_seq_len:
            q.append(0)
    
    a_labels = []
    for a in ans_dic.data:
        a_labels.append(ans_dic.to_id(a))
        # print (ans_dic.to_id(a))
    
    # BERT input embedding
    input_ids = q_tokens
    input_segment_ids = [[0]*max_seq_len for i in range(len(question_dic))]         # token_type_ids # segment_ids存儲的是句子的id，id為0就是第一句，id為1就是第二句
    input_masks = []                                                                # position_ids:1代表是真實的單詞id，0代表補全位
    for i in range(len(question_dic)):
        position_ids = []
        for j in range(original_length[i]):
            position_ids.append(1)
        while len(position_ids)<max_seq_len:
            position_ids.append(0)
        input_masks.append(position_ids)
    answer_lables = a_labels
    assert len(input_ids) == len(question_dic) and len(input_ids) == len(input_masks) and len(input_ids) == len(input_segment_ids)
    data_features = {'input_ids':input_ids,
                    'input_segment_ids':input_segment_ids,
                    'input_masks':input_masks,
                    'answer_lables':answer_lables,
                    'question_dic':question_dic,
                    'answer_dic':ans_dic}

    # 因為train_data保證可以拿到所有類別的資料，且之後要做predict時才能將預測的結果做正確轉換
    if FileName == 'train_data.txt':
        fp = open('trained_model/data_features.pkl', 'wb')
        pickle.dump(data_features, fp)
        fp.close()

    return data_features

if __name__ == "__main__":
    data_features = convert_data_to_feature('train_data.txt')
    print(data_features['input_ids'][0])
    print(data_features['input_segment_ids'][0])
    print(data_features['input_masks'][0])