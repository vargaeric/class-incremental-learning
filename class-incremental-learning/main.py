from icarl import iCaRL


def main():
    print('Class incremental learning application have started!')

    icarl = iCaRL()

    icarl.train()


if __name__ == '__main__':
    main()
