import argparse 
import sys
import matplotlib.pyplot as plt

def main(argv):

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "log_file", default='nohup_1_cls_12_28.txt',
        help = "path to log file"
        )

    args = parser.parse_args()

    f = open(args.log_file)
    
    lines  = [line.rstrip("\n") for line in f.readlines()]
    
    numbers = {'1','2','3','4','5','6','7','8','9'}

    iters = []
    loss = []
    
    fig,ax = plt.subplots()

    prev_line = ""
    for line in lines:
        args = line.split(' ')
        if args[0][-1:]==':' and args[0][0] in numbers :
            iters.append(int(args[0][:-1]))
            f = ''
            for i in args[2]:
                if i == 'S':
                    break
                else:
                    f += i
            
            loss.append(float(f))
    
    ax.plot(iters,loss)
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.grid()
	
    ticks = range(0,250,10)
    
    #ax.set_yticks(ticks)
    plt.savefig('loss.png')
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv)
