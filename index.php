<?php

declare(strict_types=1);

use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;

require_once('vendor/autoload.php');

$csv = new CSV('datasets/cars.csv', true);
$dataset = Labeled::fromIterator($csv);
$dataset->apply(new NumericStringConverter());

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$tree = new ClassificationTree();
$tree->train($training);

$predictions = $tree->predict($testing);

$confusionMatrix = new ConfusionMatrix();

$accuracy = new Accuracy();

$accuracy_percentage = number_format($accuracy->score($predictions, $testing->labels())*100,0);

echo "Processed ".$dataset->numSamples()." items with an accuracy of $accuracy_percentage%\n\n";

// Asking to model if this car probably would to be sold
var_dump($tree->predict(new Unlabeled([[
    '8124',
    'Hyundai',
    '2013',
    '320000',
    '110000',
    'Central',
    'California',
    'Los Angeles',
    'Petrol',
    'Individual',
    'Manual',
    'First_Owner',
    18.5,
    1197,
    82.85,
    '113.7Nm@ 4000rpm',
    5]]
)));