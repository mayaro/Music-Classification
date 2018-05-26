import { Component } from '@angular/core';
import { ClassificationService } from '../../services/classification.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'mc-classification',
  templateUrl: './classification.component.html',
  styleUrls: [ './classification.component.css' ]
})
export class ClassificationComponent {
  
  public classification$: Observable<Object>;

  constructor(
    private classificationService: ClassificationService
  ) {
    this.classificationService = classificationService;
  }

  private _url : string;

  public get url() : string {
    return this._url;
  }

  public set url(newUrl : string) {
    this._url = newUrl;
  }

  public async classify($event: KeyboardEvent) {
    console.log(this);
    this.classification$ = await this.classificationService.classify(this._url);
    console.log(this.classification$)
  }
  
} 