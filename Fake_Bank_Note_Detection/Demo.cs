using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Football_Team_Rating_Prediction
{
    internal class Demo
    {
        public static void Execute()
        {
            // Create new MLContext
            MLContext context = new MLContext();

            // Data Path
            var path = "C:\\Users\\akrc2\\Downloads\\bank_notes.csv";

            // Load Data
            var dataView = context.Data.LoadFromTextFile<InputModel>(path: path, hasHeader: true, separatorChar: ',');

            // Split the data into training and testing sets
            var dataSplit = context.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Prepare Data & Create pipeline
            var pipeline = context.Transforms.SelectColumns(
                nameof(InputModel.Variance), nameof(InputModel.Skewness), nameof(InputModel.Curtosis), nameof(InputModel.Entropy), nameof(InputModel.Target))
                .Append(context.Transforms.Concatenate("Features", nameof(InputModel.Variance), nameof(InputModel.Skewness), nameof(InputModel.Curtosis), nameof(InputModel.Entropy), nameof(InputModel.Target)))
                .Append(context.Transforms.Conversion.ConvertType("Label", nameof(InputModel.Target), Microsoft.ML.Data.DataKind.Boolean))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Train Model
            var model = pipeline.Fit(dataSplit.TrainSet);

            // Evaluate the Model
            var predictions = model.Transform(dataSplit.TestSet);
            var metrics = context.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");

            // Create Prediction Engine
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel { Variance = 3.62F, Skewness = 8.66F, Curtosis = -2.80F, Entropy = -0.44F };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { Variance = -1.39F, Skewness = 3.31F, Curtosis = -1.39F, Entropy = -1.99F };
            PrintResult(predictionEngine.Predict(input));
			
			// Save Model
            var save_path = "C:\\Users\\akrc2\\OneDrive\\Masaüstü\\ML.net - Fake Bank Note Detection\\BankNoteDetection.zip";

            using (var fileStream = new FileStream(save_path, FileMode.Create))
            {
                context.Model.Save(model, dataView.Schema, fileStream);
            }
        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.Prediction} | Score: {result.Score}");
        }
    }
}
