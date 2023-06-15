using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLnetBeginner.Football_Team_Rating_Prediction
{
    internal class InputModel
    {
        [LoadColumn(0)]
        public float Variance { get; set; }

        [LoadColumn(1)]
        public float Skewness { get; set; }

        [LoadColumn(2)]
        public float Curtosis { get; set; }

        [LoadColumn(3)]
        public float Entropy { get; set; }

        [LoadColumn(4)]
        public Single Target { get; set; }
    }
}
