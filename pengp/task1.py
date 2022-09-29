import pandas as pd
import numpy as np
df=pd.read_csv('000300.XSHG_2022-06-08_model_fusion.csv')
df3 = df.pivot_table(values=["prediction",'label'],index=['datetime','order_book_id'])
df4 = df.pivot_table(values=["prediction",'label'],index=['datetime'])

IC=[]
for row in df4.index:
    df0=df3.loc[(row,slice(None)),:]
    IC0=df0['prediction'].corr(df0['label'],method='spearman')
    IC.append(IC0)

information_c=pd.DataFrame(data=IC,index=df4.index,columns=['IC'])
better_days=information_c[information_c['IC']>0]    ##筛选表现较好的交易日
df=pd.read_csv('csi300_constituent_weight.csv',index_col=['datetime','order_book_id'])
df2 = pd.concat([df3,df],axis=1,join='inner')

index_sum=[]
ret=[]
for row in better_days.index:
    df_=df2.loc[(row,slice(None)),:]
    weight=np.array(df_['weight'])
    index=np.array(df_['prediction'])
    r=np.array(df_['label'])
    ret.append(np.dot(weight,r))
    index_sum.append(np.dot(weight,index))

index_pre=pd.concat([pd.DataFrame(data=index_sum,index=better_days.index,columns=['index_predict']),pd.DataFrame(data=ret,index=better_days.index,columns=['return'])],axis=1,join='inner')
index_pre=pd.concat([index_pre,pd.DataFrame(data=better_days,index=better_days.index,columns=['IC'])],axis=1,join='inner')

index_pre['up'][index_pre['return']>0]=1    #若涨，'up'等于1
IC_p=index_pre['index_predict'].corr(index_pre['up'],method='pearson')
IC_rank=index_pre['index_predict'].corr(index_pre['up'],method='spearman')
print('pearson IC',IC_p,'rank IC',IC_rank)

hreturn=index_pre[index_pre['return']>0]    #涨
lreturn=index_pre[index_pre['return']<=0]   #跌
IC_h=hreturn['return'].corr(hreturn['index_predict'],method='pearson')
IC_l=lreturn['return'].corr(lreturn['index_predict'],method='pearson')
IC_total=index_pre['return'].corr(index_pre['index_predict'],method='pearson')
print(IC_h,IC_l,IC_total)

print(index_pre['return'].describe())

import matplotlib.pyplot as plt
import seaborn as sns
sns.catplot(kind="box",y='return',data=index_pre)   #绘制箱线图
plt.show()

mean=index_pre['index_predict'].mean()      #计算占比
avg=index_pre[index_pre['index_predict']>mean]
percentage=len(avg[index_pre['return']>0])/len(avg)
print('percentage')

# index_sum_h=[]
# index_sum_l=[]
# for row in better_days.index:
#     df_1=df2.loc[(row,slice(None)),:]
#     df_1=df_1.sort_values(by=['prediction'],ascending=False)
#     df_2 = df_1.sort_values(by=['prediction'])
#     df_1=df_1.iloc[:100,:]
#     df_2 = df_2.iloc[:100, :]
#     index_sum_h.append(np.dot(np.array(df_1['weight']),np.array(df_1['prediction'])))
#     index_sum_l.append(np.dot(np.array(df_2['weight']), np.array(df_2['prediction'])))
