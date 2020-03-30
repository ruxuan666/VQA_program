#根据QA提取字典
import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict


def make_vocab_questions(input_dir):
    """Make dictionary for questions and save them into text file."""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')#匹配任意非unicode字符1次以上，“unicode：类似[^A-Za-z0-9_]”
    question_length = []
    datasets = os.listdir(input_dir) # 4个问题文件
    for dataset in datasets: #对于每个文件
        with open(input_dir+'/'+dataset) as f:
            questions = json.load(f)['questions']#[{图片id,问题，问题id},],问题编号在图片编号后补充了3位
        set_question_length = [None]*len(questions)
        for iquestion, question in enumerate(questions):#对于每个问题
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower())#将问题变为小写并切分成单词
            words = [w.strip() for w in words if len(w.strip()) > 0] #去空的情况
            vocab_set.update(words) #更新单词集
            set_question_length[iquestion] = len(words)#该句问题的长度
        question_length += set_question_length #两列表相加

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')
    
    with open('../datasets/vocab_questions.txt', 'w') as f:
        f.writelines([w+'\n' for w in vocab_list])#将单词逐行写入
    
    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))#问题最大长度


def make_vocab_answers(input_dir, n_answers):
    """Make dictionary for top n answers and save them into text file."""
    answers = defaultdict(lambda: 0) #为字典对象不存在的key自动复制0
    datasets = os.listdir(input_dir)
    for dataset in datasets: #2个回答文件
        with open(input_dir+'/'+dataset) as f:
            annotations = json.load(f)['annotations']#[{问题类型，多选回答，answers:[10个回答字典],图片编号，回答类型(number,yes/no.other)，问题编号},]
        for annotation in annotations:#对于每张图片每个问题的回答
            for answer in annotation['answers']:#[{answer,answer_confidence,answer_id(1-10)},]
                word = answer['answer'] # 锁定答案
                if re.search(r"[^\w\s]", word):#若找到非单词字符、非不可见字符，不添加
                    continue
                answers[word] += 1#回答都是一个词
                
    answers = sorted(answers, key=answers.get, reverse=True)#按照value降序排列key
    assert('<unk>' not in answers) #为真正常执行
    #前1000个高频词
    top_answers = ['<unk>'] + answers[:n_answers-1] #eg '-1' is due to '<unk>'
    
    with open('../datasets/vocab_answers.txt', 'w') as f:
        f.writelines([w+'\n' for w in top_answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))
    print('Keep top %d answers into vocab' % n_answers)


def main(args):
    input_dir = args.input_dir #../datasets
    n_answers = args.n_answers
    make_vocab_questions(input_dir+'/Questions') #包含4个json文件
    make_vocab_answers(input_dir+'/Annotations', n_answers) #包含2个json文件


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA',
                        help='directory for input questions and answers')
    parser.add_argument('--n_answers', type=int, default=1000,
                        help='the number of answers to be kept in vocab')
    args = parser.parse_args()
    main(args)