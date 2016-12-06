package dataprepare;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.io.OutputFormat;
import org.dom4j.io.SAXReader;
import org.dom4j.io.XMLWriter;

/**
 * Dom工具类
 * 
 * @author Fandy Wang(lfwang@ir.hit.edu.cn)
 * @date 2010.04.24
 * @version 0.1
 */
public class DomUtil {

  /**
   * 把一个文件解析成DOM对象
   * 
   * @param xmlName
   *          xml文件名
   * @return
   */
  public static Document parseDom(File xmlName) {
    // 使用SAXReader解析XML文档,SAXReader包含在org.dom4j.io包中。
    // inputXml是由xml文件创建的java.io.File。
    SAXReader saxReader = new SAXReader();
    saxReader.setEncoding("UTF-8");
    try {
      return saxReader.read(xmlName);
    } catch (DocumentException e) {
      e.printStackTrace();
    }
    return null;
  }
  
  /**
   * 把一个xml字符串解析成DOM对象
   * 
   * @param xmlStr
   *          xml字符串
   * @return
   */
  public static Document parseDom(String xmlStr) {
    SAXReader saxReader = new SAXReader();
    saxReader.setEncoding("UTF-8");
    try {
      return saxReader.read(new ByteArrayInputStream(xmlStr.getBytes("UTF-8")));
    } catch (DocumentException e) {
      e.printStackTrace();
    } catch (UnsupportedEncodingException e) {
      e.printStackTrace();
    }
    return null;
  }

  /**
   * 把DOM对象doc写出到文件中
   * 
   * @param filename
   *         写出的文件路径
   * @param doc
   *         欲写入的DOM
   */
  public static void writeOut(String filename, Document doc) {
    // 输出XML文档
    try {
      OutputFormat outFmt = new OutputFormat("\t", true);
      outFmt.setEncoding("UTF-8");

      XMLWriter output = new XMLWriter(new FileOutputStream(filename), outFmt);
      output.write(doc);
      output.close();
    } catch (IOException e) {
      System.out.println(e.getMessage());
    }

  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    // TODO Auto-generated method stub
    Document doc = DomUtil.parseDom(new File("kaiqi_themes.xml"));
    System.out.println(doc.asXML());
  }

}
