using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Google.Cloud.Translation.V2;
using System.IO;

namespace Transliteration
{
    class Program
    {
        static void Main(string[] args)
        {
            //These variables are to be set at runtime. Location for cred.json ; location for pahse 1 cleaned messages, location for Transliterated messages
            var credLocation = @"<GoogleTransalteAPICredential>\<Credentials>.json";
            var messageLocation = @"<Cleaned Messages Phase 1>\messages.csv";
            var fileOut = @"<Translated Messages Location>\transmessages.csv";
            try
            {
                SetupEnvironment(credLocation);
            }
            catch(Exception ex)
            {
                Console.WriteLine($"Exception has occured : {ex.Message}");
            }

            int readCount = 0;
            List<string> transMessages = new List<string>();
            using (var reader = new StreamReader(messageLocation))
            {
                while (!reader.EndOfStream)
                {
                    readCount++;
                    transMessages.Add(TransliterateText(reader.ReadLine()));
                    if(transMessages.Count() % 100 == 0)
                        Console.WriteLine($"Read {transMessages.Count()} messages");
                }
                Console.WriteLine($"Total messages read = {readCount}, Total Transliterated = {transMessages.Count}");
            }

            Console.WriteLine($"Final read messages = {readCount}, Total Transliterated = {transMessages.Count}");
            using (var fileStream = File.CreateText(fileOut))
            {
                for (int i = 0; i < transMessages.Count(); i++)
                {
                    try
                    {
                        fileStream.WriteLine(transMessages[i]);
                    }
                    catch(Exception ex)
                    {
                        Console.WriteLine("Writer exception caught");
                    }
                }
               
            }
        }

        public static string TransliterateText(string message)
        {
            var client = TranslationClient.Create();
            var convStep1 = client.TranslateText(string.IsNullOrWhiteSpace(message)?"No message": message, LanguageCodes.Hindi, LanguageCodes.English);
            var finalTrans = client.TranslateText(convStep1.TranslatedText, LanguageCodes.English, LanguageCodes.Hindi);
            return finalTrans.TranslatedText;
        }
        public static void SetupEnvironment(string location)
        {

            Environment.SetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", @"<GoogleTransalteAPICredential>\<Credentials>.json");
            Console.WriteLine($"Google app creds {Environment.GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS")}");
        }

    }
}
