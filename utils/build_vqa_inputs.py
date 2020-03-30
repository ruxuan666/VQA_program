#整合数据
import numpy as np
import json
import os
import argparse
import text_helper as text_processing
from collections import defaultdict


def extract_answers(q_answers, valid_answer_set):
    """
    :param q_answers: 10个回答字典组成得到回答列表
    :param valid_answer_set: 根据词频选择的top答案
    :return: 全部的回答词语列表，在top中的答案形成的有效答案列表
    """
    all_answers = [answer["answer"] for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)#image_set对应具体的数据
    if image_set in ['train2014', 'val2014']: #存在答案
        load_answer = True
        with open(annotation_file % image_set) as f:
            annotations = json.load(f)['annotations'] #列表，每个元素是字典，对应一条Image_QA实例
            #建立问题编号与解释字典间的字典
            qid2ann_dict = {ann['question_id']: ann for ann in annotations}
    else:
        load_answer = False
    with open(question_file % image_set) as f:
        questions = json.load(f)['questions'] #列表。每个元素为字典，对应每个Image_Q
    coco_set_name = image_set.replace('-dev', '')
    #绝对路径
    abs_image_dir = os.path.abspath(image_dir % coco_set_name)#test-dev2015的Q仍对应图像文件夹test2015
    image_name_template = 'COCO_'+coco_set_name+'_%012d'#图像名字模板
    dataset = [None]*len(questions)#全部问题数目
    
    unk_ans_count = 0
    for n_q, q in enumerate(questions):#对于每一问题
        if (n_q+1) % 10000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['image_id']#与问题对应的图像id
        question_id = q['question_id']#问题id
        image_name = image_name_template % image_id #图像文件名
        image_path = os.path.join(abs_image_dir, image_name+'.jpg')#图像路径
        question_str = q['question'] #问题文本
        question_tokens = text_processing.tokenize(question_str)#返回句子切词列表

        #构建信息字典
        iminfo = dict(image_name=image_name,
                      image_path=image_path,
                      question_id=question_id,
                      question_str=question_str,
                      question_tokens=question_tokens)
        
        if load_answer:#如果有回答
            ann = qid2ann_dict[question_id]#通过问题id对应到解释字典
            #抽取回答集合
            all_answers, valid_answers = extract_answers(ann['answers'], valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1 #没有有效回答的例子数目
            iminfo['all_answers'] = all_answers #往信息表中添加答案信息
            iminfo['valid_answers'] = valid_answers
            
        dataset[n_q] = iminfo #根据问题文件序列存储信息
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
    return dataset #返回整合后的数据集，列表型，元素为字典


def main(args):
    #设定图像路径、答案路径、问题路径
    image_dir = args.input_dir+'/Resized_Images/%s/'
    annotation_file = args.input_dir+'/Annotations/v2_mscoco_%s_annotations.json'
    question_file = args.input_dir+'/Questions/v2_OpenEnded_mscoco_%s_questions.json'

    vocab_answer_file = args.output_dir+'/vocab_answers.txt'
    answer_dict = text_processing.VocabDict(vocab_answer_file)#建立类
    valid_answer_set = set(answer_dict.word_list)  #载入文件得到单词列表并将其作为有效的回答集合

    #分别对4类文件夹做处理
    train = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'train2014')
    valid = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'val2014')
    test = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test2015')
    test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')

    #保存整合后的数据集到npy文件
    np.save(args.output_dir+'/train.npy', np.array(train))
    np.save(args.output_dir+'/valid.npy', np.array(valid))
    np.save(args.output_dir+'/train_valid.npy', np.array(train+valid))
    np.save(args.output_dir+'/test.npy', np.array(test))
    np.save(args.output_dir+'/test-dev.npy', np.array(test_dev))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='../datasets',
                        help='directory for outputs')
    
    args = parser.parse_args()

    main(args)
