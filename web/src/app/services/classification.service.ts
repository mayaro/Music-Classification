import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ClassificationModel } from '../models/classification.model';

@Injectable({
  providedIn: 'root',
})
export class ClassificationService {

  constructor(
    private http: HttpClient
  ) { }

  /**
   * Classify a song from YouTube in genres by it's URL.
   * @async
   * @param url The YouTube URL for the song to be classified
   */
  classify(url: string): Observable<Object> {
    
    return this.http.get<ClassificationModel>(url);

  }

}