#读取json数据
import json
"""
with open('./FM-IQA_dataset/FM-CH-QA.json', 'r',encoding='utf-8') as json_file:
    data = json_file.read()
    print(type(data))  # type(data) = 'str'
    print(data[:500])
    result = json.loads(data) #将json字符串转换为python字典
    print(type(result))#dict
    new_result = json.dumps(result, ensure_ascii=False) #将字典再转换为能正确输出中文的字符串
    print(type(new_result)) #str
    print(new_result)"""

def read_json():
    #with open('./datasets/Questions/v2_OpenEnded_mscoco_test2015_questions.json', 'r', encoding='utf-8') as json_file:
    with open('./datasets/Annotations/v2_mscoco_train2014_annotations.json', 'r', encoding='utf-8') as json_file:
        data = json_file.read()
        print(type(data))  # type(data) = 'str'
        print(data[:5000])


if __name__=="__main__":
    read_json()
