import matplotlib.pyplot as plt
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)   
        print ('Loaded %s..' %path)
    return file
	
def main():
    history=load_pickle('data/history.pkl')
    loss = history['loss']
    acc =history['acc']
    top3_acc = history["top3_acc"]
    val_loss = history['val_loss']
    val_acc = history['val_acc']
    val_top3_acc=history['val_top3_acc']
    val_acc[0]=0.125

    plt.title('Loss function')
    plt.plot(loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.show()   #画Loss function随epoch变化图

    plt.title('Top1 accuracy')
    plt.plot(acc,label='train top1_acc')
    plt.plot(val_acc,label='val top1_acc')
    plt.ylabel('top1 acc')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()   #画Top1 accuracy随epoch变化图

    plt.title('Top3 accuracy')
    plt.plot(top3_acc, label='train top3_acc')
    plt.plot(val_top3_acc, label='val top3_acc')
    plt.ylabel('top3 acc')
    plt.xlabel('epoch')
    plt.legend(loc='lower right')
    plt.show()   #画Top3 accuracy随epoch变化图
    #val_top3_acc=np.array(val_top3_acc)
    #print(np.max(val_top3_acc))
    print('train max top1_acc: %f' % max(acc))
    print('val max top1_acc: %f' % max(val_acc))
    print('train max top3_acc: %f' % max(top3_acc))
    print('val max top3_acc: %f' %max(val_top3_acc))
if __name__=='__main__':
    main()