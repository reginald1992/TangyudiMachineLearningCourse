#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_loan_project.py
# @Author: Shulin Liu
# @Date  : 2019/3/18
# @Desc  : 贷款申请利润最大化
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

loans_2007 = pd.read_csv("loans_2007.csv")
loans_2007.drop_duplicates()
print(loans_2007.iloc[0])
print(loans_2007.shape[1])
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
print(loans_2007.iloc[0])
print(loans_2007.shape[1])
print(loans_2007['loan_statues'].value_counts())
