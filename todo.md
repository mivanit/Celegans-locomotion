
# 2021-05-27 13:38
connection keys can't be seamlessly turned into NamedTuples because `from` is a python keyword. So, at some point I should go in and change the 'from'/'to' to something else, and then the python code can be cleaned up a bit and made type checkable