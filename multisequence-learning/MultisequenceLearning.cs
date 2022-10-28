using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;

namespace multisequence_learning
{
    public class MultiSequenceLearning
    {

        public List<double> Accuracy { get; set; }
        public List<Dictionary<string, string>>? UserPredictedValues { get; set; }
        public long ElapsedTime { get; set; }
        public string OutputPath { get; set; }
        public string RandomUserInput { get; set; }

        public MultiSequenceLearning()
        {
            Accuracy = new List<double>();
            UserPredictedValues = new List<Dictionary<string, string>>();
            OutputPath = "";
            RandomUserInput = "";
        }

        /// <summary>
        /// Generated the date in DateTime class of given range as start date YYYY/MM1/DD1 and end date as YYYY/MM2/DD2 
        /// </summary>
        /// <param name="YYYY">year of date</param>
        /// <param name="MM1">month of start date</param>
        /// <param name="MM2">month of end date</param>
        /// <param name="DD1">day of start date</param>
        /// <param name="DD2">day of end date</param>
        /// <returns></returns>
        public static DateTime GenerateRandomDate(int YYYY, int MM1, int MM2, int DD1, int DD2)
        {
            //create random input and pass on

            var random = new Random();
            DateTime startdt = new DateTime(YYYY, MM1, DD1);
            DateTime enddt = new DateTime(YYYY, MM2, DD2);
            TimeSpan timeSpan = enddt.Subtract(startdt);
            TimeSpan newSpan = new TimeSpan(random.Next(0, (int)timeSpan.TotalDays), random.Next(0, (int)timeSpan.TotalDays), 0);
            DateTime newdt = startdt.Add(newSpan);

            return newdt;
        }

        /// <summary>
        /// Starts the MultiSequenceLearning Experiment
        /// </summary>
        /// <param name="dataset">local full path for input dataset</param>
        public void StartExperiment(string dataset)
        {
            int inputBits = 100;
            int maxCycles = 35;
            int numColumns = 2048;
            string[] sequenceFormatType = { "byMonth" /* 720 */, "byWeek" /* 168 */, "byDay" /* 24 */};

            List<Dictionary<string, int[]>> encodedData;
            EncoderBase encoderDateTime;
            HtmPredictionEngine trainedEngine;
            Stopwatch sw = new Stopwatch();

            sw.Start();
            PrepareTrainingData(sequenceFormatType, dataset, out encodedData, out encoderDateTime);

            RunTraining(inputBits, maxCycles, numColumns, encodedData, encoderDateTime, out trainedEngine);

            RunPrediction(trainedEngine);
            sw.Stop();

            ElapsedTime = (sw.ElapsedMilliseconds) / 1000; //milliseconds to seconds
        }

        /// <summary>
        /// Preparing Data for Learning - 1. Created MultiSequence and 2. Encoding Input
        /// </summary>
        /// <param name="sequenceFormatType">in params for format the sequence by month or week or day</param>
        /// <param name="encodedData">out params of encoded data</param>
        /// <param name="encoderDateTime">out params for encoded datetime</param>
        private static void PrepareTrainingData(string[] sequenceFormatType, string dataset, out List<Dictionary<string, int[]>> encodedData, out EncoderBase encoderDateTime)
        {
            Console.WriteLine("Reading CSV File..");
            //var csvData = HelperMethods.ReadPowerConsumptionDataFromCSV(PowerConsumptionCSV, sequenceFormatType[0]);
            var csvData = HelperMethods.ReadPowerConsumptionDataFromCSV(dataset, sequenceFormatType[1]);
            Console.WriteLine("Completed reading CSV File..");

            Console.WriteLine("Encoding data read from CSV...");
            encodedData = HelperMethods.EncodePowerConsumptionData(csvData, true);
            encoderDateTime = HelperMethods.FetchDateTimeEncoder();
        }

        /// <summary>
        /// Getting trained model by MultiSequence Learning
        /// </summary>
        /// <param name="inputBits"></param>
        /// <param name="maxCycles"></param>
        /// <param name="numColumns"></param>
        /// <param name="encodedData"></param>
        /// <param name="encoderDateTime"></param>
        /// <param name="trainedHTMmodel"></param>
        private void RunTraining(int inputBits, int maxCycles, int numColumns, List<Dictionary<string, int[]>> encodedData, EncoderBase encoderDateTime, out HtmPredictionEngine trainedHTMmodel)
        {
            Console.WriteLine("Started Learning...");
            /*
             * Running MultiSequence Learning experiment here
             */
            trainedHTMmodel = Run(inputBits, maxCycles, numColumns, encoderDateTime, encodedData);

            Console.WriteLine("Done Learning");
        }

        /// <summary>
        /// Takes user input and gives predicted label
        /// </summary>
        /// <param name="trainedEngine">trained object of class HtmPredictionEngine which will be used to predict</param>
        private void RunPrediction(HtmPredictionEngine trainedEngine)
        {
            var logs = new List<String>();
            //Random generated user date;
            var userInput = GenerateRandomDate(2010, 07, 12, 10, 25);
            RandomUserInput = userInput.ToString();
            Console.WriteLine($"Random User Input Date: {userInput.ToString()}");
            logs.Add($"Random User Input Date: {userInput.ToString()}");
            Dictionary<string, string> pVal;
            var sdr = HelperMethods.EncodeSingleInput(userInput.ToString());
            trainedEngine.Reset();
            var predictedValuesForUserInput = trainedEngine.Predict(sdr);
            if (predictedValuesForUserInput.Count > 0)
            {
                foreach (var predictedVal in predictedValuesForUserInput)
                {
                    pVal = new Dictionary<string, string>();
                    Console.WriteLine($"SIMILARITY: {predictedVal.Similarity} PREDICTED VALUE: {predictedVal.PredictedInput}");
                    logs.Add($"SIMILARITY: {predictedVal.Similarity} PREDICTED VALUE: {predictedVal.PredictedInput}");
                    pVal.Add(predictedVal.Similarity.ToString(), predictedVal.PredictedInput);
                    //MultiSequenceLearning.PredictedValues.Add(pVal);
                    UserPredictedValues?.Add(pVal);

                }
            }
            else
            {
                Console.WriteLine("Nothing predicted :(");
                logs.Add("Nothing predicted :(");
            }

            File.AppendAllLines(OutputPath, logs);

        }

        /// <summary>
        /// Multi-sequence Learning MOdel is trained here
        /// </summary>
        /// <param name="inputBits"></param>
        /// <param name="maxCycles"></param>
        /// <param name="numColumns"></param>
        /// <param name="encoder"></param>
        /// <param name="sequences">Multi Sequence for training</param>
        /// <returns>Learned CortexLayer and HtmClassifier for prediction</returns>
        //public Dictionary<CortexLayer<object,object>, HtmClassifier<string, ComputeCycle>> Run(int inputBits, int maxCycles, int numColumns, EncoderBase encoder, List<Dictionary<string,int[]>> sequences)
        public HtmPredictionEngine Run(int inputBits, int maxCycles, int numColumns, EncoderBase encoder, List<Dictionary<string, int[]>> sequences)
        {
            /* HTM Config */
            var htmConfig = HelperMethods.FetchHTMConfig(inputBits, numColumns);

            /* Creating Connections */
            var mem = new Connections(htmConfig);

            /* Getting HTM CLassifier */
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();

            /* Get Cortex Layer */
            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            /* HPA Stable Flag */
            bool isInStableState = false;

            /* Learn Flag */
            bool learn = true;

            /* Number of new born cycles */
            int newbornCycle = 0;

            /* Logs */
            var OUTPUT_LOG_LIST = new List<Dictionary<int, string>>();
            var OUTPUT_LOG = new Dictionary<int, string>();
            var OUTPUT_trainingAccuracy_graph = new List<Dictionary<int, double>>();

            /* Minimum Cycles */
            int numUniqueInputs = sequences.Count;

            // For more information see following paper: https://www.scitepress.org/Papers/2021/103142/103142.pdf
            HomeostaticPlasticityController hpc = new HomeostaticPlasticityController(mem, numUniqueInputs, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal SP should never enter unstable state after stable state.
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");

                // We are not learning in instable state.
                learn = isInStableState = isStable;

            }, numOfCyclesToWaitOnChange: 50);


            /* Spatial Pooler with HomeoPlasticityController using Connections */
            SpatialPoolerMT sp = new SpatialPoolerMT();
            sp.Init(mem);

            /* Temporal Memory with Connections */
            TemporalMemory tm = new TemporalMemory();
            tm.Init(mem);

            /* Adding Encoder to Cortex Layer */
            //layer1.HtmModules.Add("encoder", encoder); /* not needed since encoded already */

            /* Adding Spatial Pooler to Cortex Layer */
            layer1.HtmModules.Add("sp", sp);

            // Container for Previous Active Columns
            int[] prevActiveCols = new int[0];

            int computeCycle = 0;
            int maxComputeCycles = maxCycles;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            /*
             * Training SP to get stable. New-born stage.
             */

            /* Stable Condition Loop --- Loop 1 */
            for (int i = 0; i < maxComputeCycles && isInStableState == false; i++)
            {
                computeCycle++;
                newbornCycle++;
                Debug.WriteLine($"-------------- Newborn Cycle {newbornCycle} ---------------");
                Console.WriteLine($"-------------- Training SP Newborn Cycle {newbornCycle} ---------------");

                /* For each sequence in multi-sequence --- Loop 2 */
                foreach (var sequence in sequences)
                {
                    /* For each element (dictionary) in sequence --- Loop 3 */
                    foreach (var element in sequence)
                    {
                        string[] splitKeyv = element.Key.Split(",");
                        var observationClass = splitKeyv[0]; // OBSERVATION LABEL || SEQUENCE LABEL
                        var elementSDR = element.Value; // ELEMENT IN ONE SEQUENCE

                        Debug.WriteLine($"-------------- {observationClass} ---------------");

                        var lyrOut = layer1.Compute(elementSDR, true);     /* CORTEX LAYER OUTPUT with elementSDR as INPUT and LEARN = TRUE */
                        //var lyrOut = layer1.Compute(elementSDR, learn);    /* CORTEX LAYER OUTPUT with elementSDR as INPUT and LEARN = if TRUE */

                        if (isInStableState)
                            break;
                    }

                    if (isInStableState)
                        break;
                }
            }

            // Clear all learned patterns in the classifier.
            //cls.ClearState();

            // We activate here the Temporal Memory algorithm.
            layer1.HtmModules.Add("tm", tm);

            string lastPredictedValue = "-1";
            List<string> lastPredictedValueList = new List<string>();
            double lastCycleAccuracy = 0;
            double accuracy = 0;

            List<List<string>> possibleSequence = new List<List<string>>();

            /* Training SP+TM together */
            /* For each sequence in multi-sequence --- Loop 1 */
            foreach (var sequence in sequences)
            {
                int SequencesMatchCount = 0; // NUMBER OF MATCHES
                var tempLOGFILE = new Dictionary<int, string>();
                var tempLOGGRAPH = new Dictionary<int, double>();
                double SaturatedAccuracyCount = 0;

                /* Loop until maxCycles --- Loop 2*/
                for (int i = 0; i < maxCycles; i++)
                {
                    Console.WriteLine($"-------------- Training SP+TM Newborn Cycle {i} ---------------");
                    List<string> ElementWiseClasses = new List<string>();

                    /* Element in sequenc match counter */
                    int elementMatches = 0;

                    /* For each element (dictionary) in sequence --- Loop 3 */
                    foreach (var Elements in sequence)
                    {

                        // key,value = 21.2W,SDR(timesegment) 
                        string[] splitKey = Elements.Key.Split(",");
                        var observationLabel = splitKey[0];

                        var lyrOut = new ComputeCycle();

                        /* Get Compute Cycle */
                        //Compute(key)  <= this should be correct
                        lyrOut = layer1.Compute(Elements.Value, learn) as ComputeCycle;
                        Debug.WriteLine(string.Join(',', lyrOut.ActivColumnIndicies));

                        /* Get Active Cells */
                        List<Cell> actCells = (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count) ? lyrOut.ActiveCells : lyrOut.WinnerCells;

                        /* Learn the combination of Label and Active Cells   power = 21.2 watts */
                        cls.Learn(observationLabel, actCells.ToArray());

                        if (lastPredictedValue == observationLabel && lastPredictedValue != "")
                        {
                            elementMatches++;
                            Debug.WriteLine($"Match. Actual value: {observationLabel} - Predicted value: {lastPredictedValue}");
                        }
                        else
                        {
                            Debug.WriteLine($"Mismatch! Actual value: {observationLabel} - Predicted values: {lastPredictedValue}");
                        }

                        Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(actCells.Select(c => c.Index).ToArray())}");

                        if (learn == false)
                            Debug.WriteLine($"Inference mode");

                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            var predictedInputValue = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);

                            Debug.WriteLine($"Current Input: {observationLabel}");
                            Debug.WriteLine("The predictions with similarity greater than 50% are");

                            foreach (var t in predictedInputValue)
                            {

                                if (t.Similarity >= (double)50.00)
                                {
                                    Debug.WriteLine($"Predicted Input: {string.Join(", ", t.PredictedInput)},\tSimilarity Percentage: {string.Join(", ", t.Similarity)}, \tNumber of Same Bits: {string.Join(", ", t.NumOfSameBits)}");
                                }
                            }

                            lastPredictedValue = predictedInputValue.First().PredictedInput;

                        }
                    }

                    double maxPossibleAccuraccy = (double)((double)sequence.Count - 1) / (double)sequence.Count * 100.0;

                    accuracy = (double)elementMatches / (double)sequence.Count * 100.0;

                    Debug.WriteLine($"Cycle : {i} \t Accuracy:{accuracy}");
                    tempLOGGRAPH.Add(i, accuracy);
                    if (accuracy >= maxPossibleAccuraccy)
                    {
                        SequencesMatchCount++;
                        Debug.WriteLine($"100% accuracy reched {SequencesMatchCount} times.");
                        Console.WriteLine($"100% accuracy reched {SequencesMatchCount} times.");
                        tempLOGFILE.Add(i, $"Cycle : {i} \t  Accuracy:{accuracy} as 100% \t Number of times repeated {SequencesMatchCount}");
                        Accuracy.Add(accuracy);
                        if (SequencesMatchCount >= 30)
                        {
                            SaturatedAccuracyCount++;
                            tempLOGFILE.Add(i, $"Cycle : {i} \t  SaturatedAccuracyCount : {SaturatedAccuracyCount} \t SequenceMatchCount : {SequencesMatchCount} >= 30 breaking..");
                            break;
                        }
                    }
                    else if (SequencesMatchCount >= 0)
                    {
                        SaturatedAccuracyCount = 0;
                        SequencesMatchCount = 0;
                        lastCycleAccuracy = accuracy;
                        tempLOGFILE.Add(i, $"Cycle : {i} \t Accuracy :{accuracy} \t ");
                        Accuracy.Add(accuracy);
                    }
                    lastPredictedValueList.Clear();

                }

                tm.Reset(mem);
                learn = true;
                OUTPUT_LOG_LIST.Add(tempLOGFILE);

            }

            sw.Stop();

            TimeSpan timeSpan = sw.Elapsed;

            //****************DISPLAY STATUS OF EXPERIMENT
            Debug.WriteLine("-------------------TRAINING END------------------------");
            Console.WriteLine("-----------------TRAINING END------------------------");
            string timespend = $"Training Time : {timeSpan.TotalMinutes} total minutes and {timeSpan.Seconds} seconds";
            Console.WriteLine(timespend);
            Debug.WriteLine("-------------------WRTING TRAINING OUTPUT LOGS---------------------");
            Console.WriteLine("-------------------WRTING TRAINING OUTPUT LOGS------------------------");
            //*****************

            DateTime now = DateTime.Now;
            string filename = now.ToString("g");
            // remove any / or : or -
            filename = filename.Replace("/", "");
            filename = filename.Replace("-", "");
            filename = filename.Replace(":", "");
            filename = $"PowerConsumptionPredictionExperiment_{filename.Split(" ")[0]}_{now.Ticks.ToString()}.txt";
            string path = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, filename));
            OutputPath = path;
            using (StreamWriter swOutput = File.CreateText(OutputPath))
            {
                swOutput.WriteLine($"{filename}");
                foreach (var SequencelogCycle in OUTPUT_LOG_LIST)
                {
                    swOutput.WriteLine("******Sequence Starting*****");
                    foreach (var cycleOutPutLog in SequencelogCycle)
                    {
                        swOutput.WriteLine(cycleOutPutLog.Value, true);
                    }
                    swOutput.WriteLine("****Sequence Ending*****");

                }
            }
            File.AppendAllText(OutputPath, $"{timespend} \n");
            Debug.WriteLine("-------------------TRAINING LOGS HAS BEEN CREATED---------------------");
            Console.WriteLine("-------------------TRAINING LOGS HAS BEEN CREATED------------------------");

            return new HtmPredictionEngine { Layer = layer1, Classifier = cls, Connections = mem };
        }

        public class HtmPredictionEngine
        {
            public void Reset()
            {
                var tm = this.Layer.HtmModules.FirstOrDefault(m => m.Value is TemporalMemory);
                ((TemporalMemory)tm.Value).Reset(this.Connections);
            }
            public List<ClassifierResult<string>> Predict(int[] input)
            {
                var lyrOut = this.Layer.Compute(input, false) as ComputeCycle;

                List<ClassifierResult<string>> predictedInputValues = this.Classifier.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);

                return predictedInputValues;
            }

            public Connections Connections { get; set; }

            public CortexLayer<object, object> Layer { get; set; }

            public HtmClassifier<string, ComputeCycle> Classifier { get; set; }
        }


    }
}
