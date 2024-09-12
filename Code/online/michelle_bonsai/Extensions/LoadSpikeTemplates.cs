using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Source)]
public class LoadSpikeTemplates
{
    [Description("The .dat or .bin filepath")]
    [Category("Configuration")]
    public string Sourcepath { get; set; }

    [Description("The number of samples collected for each channel per sample")]
    [Category("Configuration")]
    public int[] TemplatesToTrack { get; set; }

    public IObservable<int> Process()
    {
        return Observable.Return(0);
    }
}
