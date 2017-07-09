#!/usr/bin/python
#coding:utf-8

def compute_pre(TP,pre_true):
	return TP*100.0/pre_true

def compute_recall(TP,all_true):
	return TP*100.0/all_true

tp = 0
pre_true = 0
all_true = 0
with open('result.txt','r') as f:
	for line in f:
		lines = line.strip().split('\t')
		if(lines[3] == '4' and lines[0] == '水利环境公共设施管理'):
			tp += 1
		if(lines[3] == '4'):
			pre_true += 1
		if(lines[0] == '水利环境公共设施管理'):
			all_true += 1
	print str(compute_pre(tp,pre_true)) + '%'
	print str(compute_recall(tp,all_true)) + '%'
		
