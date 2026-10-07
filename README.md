# עברית עד 1948

מודל שפה קטן (GPT, כ־46 מיליון פרמטרים) שאומן מאפס רק על עברית שנכתבה לפני 1948, והאתר שמריץ אותו בדפדפן.
כתבו התחלה של משפט, והמודל ימשיך אותה. ההרצה כולה מקומית בדפדפן, ושום טקסט לא נשלח לשרת.

- **נתונים:** כ־17 אלף יצירות בנחלת הכלל מ[פרויקט בן־יהודה](https://benyehuda.org) (באדיבות מתנדבי הפרויקט),
  מסוננות לפי שנת הפטירה של הכותב או המתרגם, ולפי תיוג התקופה של עורכי הפרויקט.
- **אימון:** [nanoGPT](https://github.com/karpathy/nanoGPT) על RTX 4060 Laptop. 12 שכבות ברוחב 512, חלון הקשר של 512 טוקנים,
  וטוקנייזר BPE בגודל 16K (SentencePiece).
- **בדפדפן:** המודל יוצא ל־ONNX עם KV cache ודחוס ל־int8, ורץ עם [onnxruntime-web](https://onnxruntime.ai) (רישיון MIT).

---

*Hebrew before 1948:* a ~46M-parameter GPT trained from scratch only on pre-1948 Hebrew (public-domain texts from Project Ben-Yehuda), running fully in the browser via onnxruntime-web.
