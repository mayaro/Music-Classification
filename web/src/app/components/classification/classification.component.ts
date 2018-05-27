import { Component } from '@angular/core';
import { Observable, Subject, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ToastrService } from 'ngx-toastr';
import { ClassificationService } from '../../services/classification.service';
import { ClassificationModel } from '../../models/classification.model';

@Component({
  selector: 'mc-classification',
  templateUrl: './classification.component.html',
  styleUrls: [ './classification.component.css' ]
})
export class ClassificationComponent {
  
  public classification$: Observable<Object>;

  public pristine: Boolean = true;

  public loadingError$ = new Subject<Boolean>();
  
  public loadingErrorMessage: string;

  constructor(
    private classificationService: ClassificationService,
    private toastr: ToastrService
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

  public classify($event: KeyboardEvent) {
    this.pristine = false;
    this.classification$ = null;

    this.classification$ = this.classificationService.classify(this._url).pipe(
      catchError((error: any) => {
        this.toastr.error(`
          ${error.status} ${error.statusText}:
          Error classifying song. Please check console for full error.
        `, null, {positionClass: 'toast-bottom-right'});
        console.error(error);
        this.loadingError$.next(false);
        return of();
      })
    );
  }
  
} 