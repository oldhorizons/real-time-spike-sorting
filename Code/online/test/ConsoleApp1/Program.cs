using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using System.IO;
using Bonsai.Dsp;
using System.CodeDom.Compiler;

namespace Test {
    public static class Program {
        public static void Main(string[] args) {
            // Console.WriteLine("Hello world!");
            TestPlease t = new TestPlease();
        }
    }
    public class TestPlease {
        public TestPlease() {
            int[] template0 = new int[] { 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
            int[] template1 = new int[]{1, 2, 3, 4, 3, 2, 1};
            int[] template2 = new int[]{1, 3, 5, 7, 5, 3, 1};

            SpikeWaveform s0 = new SpikeWaveform
            {
                ChannelIndex = 0,
                SampleIndex = 0,
                Waveform = Mat.FromArray(template0)
            };

            SpikeWaveform s1 = new SpikeWaveform {
                ChannelIndex = 0,
                SampleIndex = 0,
                Waveform = Mat.FromArray(template1)
            };

            SpikeWaveform s2 = new SpikeWaveform {
                ChannelIndex = 10,
                SampleIndex = 0,
                Waveform = Mat.FromArray(template2)
            };

            List<SpikeWaveform> inputs = new List<SpikeWaveform>();
            inputs.Add(s0);
            inputs.Add(s1);
            inputs.Add(s2);

            CompareTemplates temp = new CompareTemplates();
            List<bool> outs = temp.Process(inputs);
        }
    }

    public class CompareTemplates
    {
        public enum ComparisonMethod {
            Cosine,
            CrossCor,
            Dtw
        }

        public ComparisonMethod comparisonMethod;
        public string SourcePath { get; set; } = "C:/Users/miche/OneDrive/Documents/01 Uni/REIT4841/Data/sim_hybrid_ZFM_10sec/kilosort4/templates";
        public int[] TemplatesToTrack { get; set; } = new int[]{30};
        public int NumSamples { get; set; } = 61;
        public bool ConvertToU8 { get; set; } = false;
        public float DistanceThreshold { get; set; } = 2;
        public float SimilarityThreshold { get; set; } = 0;

        public List<bool> Process(List<SpikeWaveform> source)
        {
            List<bool> results = new List<bool>();
            List<SpikeWaveform> templates = LoadTemplates();
            Console.WriteLine("TEMPLATES LOADED");
            foreach (SpikeWaveform waveform in source) {
                foreach (SpikeWaveform template in templates) {
                    results.Add(SimilarityMeasure(waveform, template) > SimilarityThreshold);
                }
            }
            return results;
        }

        private float SimilarityMeasure(SpikeWaveform source, SpikeWaveform template) {
            Console.WriteLine("GOT ONE");
            return CosineSimilarity(source, template);
        }

        private float CosineSimilarity(SpikeWaveform source, SpikeWaveform template) {
            //if (Math.Abs(source.ChannelIndex - template.ChannelIndex) > SimilarityThreshold) {
            //    return -1f;
            //}
            AlignWaveforms(ref source, ref template);
            int max = source.Waveform.Cols;
            Console.WriteLine("Num Cols: {0}", max);
            double dotProduct = 0;
            double sMag, tMag;
            sMag = tMag = 0;
            Mat halfDot = source.Waveform * template.Waveform;
            source.Waveform *= source.Waveform;
            template.Waveform *= template.Waveform;
            for (int i = 0; i < max; i++) {
                dotProduct += halfDot[i].Val0;
                tMag += template.Waveform[i].Val0;
                sMag += source.Waveform[i].Val0;
            }
            tMag = Math.Sqrt(tMag);
            sMag = Math.Sqrt(sMag);
            return (float)(dotProduct / (tMag * sMag));
        }

        private void AlignWaveforms(ref SpikeWaveform source, ref SpikeWaveform template) {
            //initialise stuff
            double sMax, tMax, sMin, tMin;
            int sMaxI, tMaxI, sMinI, tMinI;
            Mat sWav = source.Waveform;
            Mat tWav = template.Waveform;
            int lenDiff = sWav.Cols - tWav.Cols;
            sMaxI = tMaxI = sMinI = tMinI = 0;
            sMax = sMin = sWav[0].Val0;
            tMax = tMin = tWav[0].Val0;
            for (int i = 0; i < sWav.Cols; i++) {
                if (sWav[i].Val0 > sMax) {
                    sMax = sWav[i].Val0;
                    sMaxI = i;
                } 
                else if (sWav[i].Val0 < sMin) {
                    sMin = sWav[i].Val0;
                    sMinI = i;
                }
            }
            for (int i = 0; i < tWav.Cols; i++) {
                if (tWav[i].Val0 > tMax) {
                    tMax = tWav[i].Val0;
                    tMaxI = i;
                } 
                else if (tWav[i].Val0 < tMin) {
                    tMin = tWav[i].Val0;
                    tMinI = i;
                }
            }
            //todo because both are uint8_t the min won't matter but it could be handy for later so I'm keeping it?????
            int offset = sMaxI - tMaxI;
            lenDiff -= offset;
            int sOff = offset > 0 ? offset : 0;
            int tOff = offset < 0 ? -offset : 0;
            int sCut = lenDiff > 0 ? lenDiff : 0;
            int tCut = lenDiff < 0 ? -lenDiff : 0;

            source.Waveform = ShortenMat(sWav, sOff, sCut);
            template.Waveform = ShortenMat(tWav, tOff, tCut);
        }
        
        private Mat ShortenMat(Mat source, int start, int end) {
            int newLen = source.Cols - (start + end);
            if (newLen == source.Cols) {
                return source;
            }
            double[] newWav = new double[newLen];
            for (int i = 0; i < newLen; i++) {
                newWav[i] = source[i + start].Val0;
            }
            return Mat.CreateMatHeader(newWav);
        }

        private List<SpikeWaveform> LoadTemplates()
        { 
            List<SpikeWaveform> templates = new List<SpikeWaveform>();
            for (int i = 0; i < TemplatesToTrack.Length; i++) {
                string filename = String.Format("{0}/t{1}.txt", SourcePath, TemplatesToTrack[i]); //TODO CSV
                SpikeWaveform template = GetSingleChanWaveform(filename);
                templates.Add(template);
            }
            return templates;
        }

        // Returns the matrix representation of a template as a 1-d Mat file
        // todo add a way to track multiple channels? 
        private SpikeTemplate GetSingleChanWaveform(string filename)
        {
            int numChannels = 0;
            int numSamples = 0;
            float[] chanMax = new float[1];
            int[] chanMaxIndex = new int[1];
            List<List<float>> floats = new List<List<float>>();

            using (StreamReader reader = new StreamReader(filename))
            {
                //read list out
                while (!reader.EndOfStream)
                {
                    // read line and parse to float array
                    string[] line = reader.ReadLine().Split(',');
                    List<float> floatLine = new List<float>();
                    foreach (string item in line)
                    {
                        floatLine.Add(float.Parse(item));
                    }
                    // initialise numChannels and chanMax if this is the first run through
                    if (numChannels == 0)
                    {
                        numChannels = floatLine.ToArray().Length;
                        chanMax = new float[numChannels];
                        chanMaxIndex = new int[numChannels];
                    }
                    //update channel maximums
                    for (int i = 0; i < numChannels; i++)
                    {
                        if (Math.Abs(floatLine[i]) > chanMax[i])
                        {
                            chanMaxIndex[i] = numSamples;
                            chanMax[i] = Math.Abs(floatLine[i]);
                        }
                    }
                    floats.Add(floatLine);
                    numSamples++;
                }
            }
            float maxValue = chanMax.Max();
            int maxIndex = chanMax.ToList().IndexOf(maxValue);


            // cut off the leading and trailing zeroes because you don't want to try to match a flat line
            int offset = 0;
            int end = numSamples;
            for (int i = 0; i < numSamples; i++)
            {
                if (floats[i][maxIndex] != 0)
                {
                    break;
                }
                else
                {
                    offset++;
                }
            }
            for (int i = numSamples; i > 0; i--)
            {
                if (floats[i][maxIndex] != 0)
                {
                    break;
                }
                else
                {
                    end--;
                }
            }

            //add a LITTLE buffer of zeroes
            offset = Math.Max(0, offset - 2);
            end = Math.Min(numSamples, end + 2);

            // get the channel info at each sample
            float[] buffer = new float[numSamples];
            for (int i = offset; i < end; i++)
            {
                buffer[i] = floats[i][maxIndex];
            }

            Mat waveform = Mat.FromArray(buffer);

            // return waveform
            return new SpikeTemplate
            {
                ChannelIndex = maxIndex,
                SampleIndex = 0,
                Waveform = waveform,
                AlignMax = buffer.Contains(maxValue)
            };
        }

        protected class SpikeTemplate : SpikeWaveform
        {
            public bool AlignMax { get; set; }
        }
    }
}
