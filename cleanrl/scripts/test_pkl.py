import sys
import pickle as pkl

if __name__ == "__main__":
  data = pkl.load(open(sys.argv[1],'rb'))
  print(data)
