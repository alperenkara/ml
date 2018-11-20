from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T) 
# transpose (30,455) 30 samples, 455 features
# with transpose (455,30)
print("testing accuracy {}".format(lr.score(x_test.T,y_test.T)))
