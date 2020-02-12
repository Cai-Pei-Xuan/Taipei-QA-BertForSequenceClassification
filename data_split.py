# -*-coding:UTF-8 -*-

# 存檔
def save_data(FileName, data_list):
    fp = open(FileName, "w", encoding='utf-8')
    for data in data_list:
        fp.write(data)
        fp.write('\n')
    fp.close()

# 資料分割，因為有149個類別，所以為了公正，要確保切割時，train_data要有每個類別的資料
def data_split():
    with open('Taipei_QA_new.txt','r',encoding='utf-8') as f:
        data = f.read()
    qa_pairs = data.split("\n")

    before = ""
    now = ""
    count = 0
    train_data = []
    test_data = []
    valid_data = []
    for qa_pair in qa_pairs:
        try:
            a,q = qa_pair.split()
            now = a

            if before != now:
                count = 0
            count += 1
            if count % 5 == 3:
                test_data.append(qa_pair)
            elif count % 5 == 4:
                valid_data.append(qa_pair)
            else:
                train_data.append(qa_pair)

            before = now
        except:
            continue

    save_data("train_data.txt", train_data)
    save_data("test_data.txt", test_data)
    save_data("valid_data.txt", valid_data)

if __name__ == "__main__":
   data_split()