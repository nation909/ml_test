import pandas as pd

#  converters={"nounsAllIndex": lambda x: x.strip("[]").replace("'", "").split(", ")}
call_result = pd.read_csv('../dataset/calldata_csv/20190329/call_result.csv', encoding='euc-kr', delimiter=',',
                          converters={"nounsAllIndex": lambda x: x.strip("()").replace(",", ":")})
print("call_result.head(5): ", call_result.head(5))
print("call_result.info(): ", call_result.info())
nounsIndex = call_result['nounsAllIndex']
print("nounsIndex: ", type(nounsIndex), nounsIndex[:5])
# for i in nounsIndex:
#     print("i: ", type(i), i)
