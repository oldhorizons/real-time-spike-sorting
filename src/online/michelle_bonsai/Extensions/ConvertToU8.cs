using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ConvertToU8
{
    public IObservable<Mat> Process(IObservable<Mat> source)
    {
        return source.Select(value => {
            Mat mat2 = new Mat(value.Rows, value.Cols, Depth.U8, 1);
            CV.ConvertScaleAbs(value, mat2, 0.0000425);
            return mat2;
        });
    }
}
