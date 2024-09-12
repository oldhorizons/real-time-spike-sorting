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
public class SpikeDetect
{
    public IObservable<Mat> Process(IObservable<Mat> source)
    {
        return source.Select(value => value);
    }
}
