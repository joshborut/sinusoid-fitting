from dataset_generator import get_target, get_dataset, scatter_plot_xy, scatter_plot_two_datasets

if __name__ == '__main__':
    dataset_size = 1000
    examples, labels = get_dataset(3, 2, dataset_size)

    training_set_ratio = 0.8
    training_set_size = int(dataset_size * training_set_ratio)
    train_examples = examples[:training_set_size]
    train_labels = labels[:training_set_size]
    test_examples = examples[training_set_size:]
    test_labels = labels[training_set_size:]

    scatter_plot_two_datasets(train_examples, train_labels, test_examples, test_labels)


